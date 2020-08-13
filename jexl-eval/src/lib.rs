/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! A JEXL evaluator written in Rust
//! This crate depends on a JEXL parser crate that handles all the parsing
//! and is a part of the same workspace.
//! JEXL is an expression language used by Mozilla, you can find more information here: https://github.com/mozilla/mozjexl
//!
//! # How to use
//! The access point for this crate is the `eval` functions of the Evaluator Struct
//! You can use the `eval` function directly to evaluate standalone statements
//!
//! For example:
//! ```rust
//! use jexl_eval::Evaluator;
//! use serde_json::json as value;
//! let evaluator = Evaluator::new();
//! assert_eq!(evaluator.eval("'Hello ' + 'World'").unwrap(), value!("Hello World"));
//! ```
//!
//! You can also run the statements against a context using the `eval_in_context` function
//! The context can be any type that implements the `serde::Serializable` trait
//! and the function will return errors if the statement doesn't match the context
//!
//! For example:
//! ```rust
//! use jexl_eval::Evaluator;
//! use serde_json::json as value;
//! let context = value!({"a": {"b": 2.0}});
//! let evaluator = Evaluator::new();
//! assert_eq!(evaluator.eval_in_context("a.b", context).unwrap(), value!(2.0));
//! ```
//!

use jexl_parser::{
    ast::{Expression, OpCode},
    Parser,
};
use serde_json::{json as value, Value};

pub mod error;
use error::*;
use std::collections::HashMap;

const EPSILON: f64 = 0.000001f64;

trait Truthy {
    fn is_truthy(&self) -> bool;

    fn is_falsey(&self) -> bool {
        !self.is_truthy()
    }
}

impl Truthy for Value {
    fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Null => true,
            Value::Number(f) => f.as_f64().unwrap() != 0.0,
            Value::String(s) => !s.is_empty(),
            // It would be better if these depended on the contents of the
            // object (empty array/object is falsey, non-empty is truthy, like
            // in Python) but this matches JS semantics. Is it worth changing?
            Value::Array(_) => true,
            Value::Object(_) => true,
        }
    }
}

type Context = Value;

/// TransformFn represents an arbitrary transform function
/// Transform functions take an arbitrary number of `serde_json::Value`to represent their arguments
/// and return a `serde_json::Value`.
/// the transform function itself is responsible for checking if the format and number of
/// the arguments is correct
///
/// Returns a Result with an `anyhow::Error`. This allows consumers to return their own custom errors
/// in the closure, and use `.into` to convert it into an `anyhow::Error`. The error message will be perserved
pub type TransformFn<'a> = Box<dyn Fn(&[Value]) -> Result<Value, anyhow::Error> + 'a>;

#[derive(Default)]
pub struct Evaluator<'a> {
    transforms: HashMap<String, TransformFn<'a>>,
}

impl<'a> Evaluator<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a custom transform function
    /// This is meant as a way to allow consumers to add their own custom functionality
    /// to the expression language.
    /// Note that the name added here has to match with
    /// the name that the transform will have when it's a part of the expression statement
    ///
    /// # Arguments:
    /// - `name`: The name of the transfrom
    /// - `transform`: The actual function. A closure the implements Fn(&[serde_json::Value]) -> Result<Value, anyhow::Error>
    ///
    /// # Example:
    ///
    /// ```rust
    /// use jexl_eval::Evaluator;
    /// use serde_json::{json as value, Value};
    ///
    /// let mut evaluator = Evaluator::new().with_transform("lower", |v: &[Value]| {
    ///    let s = v
    ///            .first()
    ///            .expect("Should have 1 argument!")
    ///            .as_str()
    ///            .expect("Should be a string!");
    ///       Ok(value!(s.to_lowercase()))
    ///  });
    ///
    /// assert_eq!(evaluator.eval("'JOHN DOe'|lower").unwrap(), value!("john doe"))
    /// ```
    pub fn with_transform<F>(mut self, name: &str, transform: F) -> Self
    where
        F: Fn(&[Value]) -> Result<Value, anyhow::Error> + 'a,
    {
        self.transforms
            .insert(name.to_string(), Box::new(transform));
        self
    }

    pub fn eval<'b>(&self, input: &'b str) -> Result<'b, Value> {
        let context = value!({});
        self.eval_in_context(input, &context)
    }

    pub fn eval_in_context<'b, T: serde::Serialize>(
        &self,
        input: &'b str,
        context: T,
    ) -> Result<'b, Value> {
        let tree = Parser::parse(input)?;
        let context = serde_json::to_value(context)?;
        if !context.is_object() {
            return Err(EvaluationError::InvalidContext);
        }
        self.eval_ast(tree, &context)
    }

    fn eval_ast<'b>(&self, ast: Expression, context: &Context) -> Result<'b, Value> {
        match ast {
            Expression::Number(n) => Ok(value!(n)),
            Expression::Boolean(b) => Ok(value!(b)),
            Expression::String(s) => Ok(value!(s)),
            Expression::Array(xs) => xs.into_iter().map(|x| self.eval_ast(*x, context)).collect(),

            Expression::Object(items) => {
                let mut map = serde_json::Map::with_capacity(items.len());
                for (key, expr) in items.into_iter() {
                    if map.contains_key(&key) {
                        return Err(EvaluationError::DuplicateObjectKey(key));
                    }
                    let value = self.eval_ast(*expr, context)?;
                    map.insert(key, value);
                }
                Ok(Value::Object(map))
            }

            Expression::IdentifierSequence(exprs) => {
                assert!(!exprs.is_empty());
                let mut rv: Option<&Value> = Some(context);
                for expr in exprs.into_iter() {
                    let key = self.eval_ast(*expr, context)?;
                    if let Some(value) = rv {
                        rv = match key {
                            Value::String(s) => value.get(&s),
                            Value::Number(f) => value.get(f.as_f64().unwrap().floor() as usize),
                            _ => return Err(EvaluationError::InvalidIndexType),
                        };
                    } else {
                        break;
                    }
                }

                Ok(rv.unwrap_or(&value!(null)).clone())
            }

            Expression::BinaryOperation {
                left,
                right,
                operation,
            } => {
                let left = self.eval_ast(*left, context)?;
                let right = self.eval_ast(*right, context)?;
                match (operation, left, right) {
                    (OpCode::And, a, b) => Ok(if a.is_truthy() { b } else { a }),
                    (OpCode::Or, a, b) => Ok(if a.is_truthy() { a } else { b }),

                    (op, Value::Number(a), Value::Number(b)) => {
                        let left = a.as_f64().unwrap();
                        let right = b.as_f64().unwrap();
                        Ok(match op {
                            OpCode::Add => value!(left + right),
                            OpCode::Subtract => value!(left - right),
                            OpCode::Multiply => value!(left * right),
                            OpCode::Divide => value!(left / right),
                            OpCode::FloorDivide => value!((left / right).floor()),
                            OpCode::Modulus => value!(left % right),
                            OpCode::Exponent => value!(left.powf(right)),
                            OpCode::Less => value!(left < right),
                            OpCode::Greater => value!(left > right),
                            OpCode::LessEqual => value!(left <= right),
                            OpCode::GreaterEqual => value!(left >= right),
                            OpCode::Equal => value!((left - right).abs() < EPSILON),
                            OpCode::NotEqual => value!((left - right).abs() > EPSILON),
                            OpCode::In => value!(false),
                            OpCode::And | OpCode::Or => {
                                unreachable!("Covered by previous case in parent match")
                            }
                        })
                    }

                    (OpCode::Add, Value::String(a), Value::String(b)) => {
                        Ok(value!(format!("{}{}", a, b)))
                    }
                    (OpCode::In, Value::String(a), Value::String(b)) => Ok(value!(b.contains(&a))),
                    (OpCode::In, left, Value::Array(v)) => Ok(value!(v.contains(&left))),
                    (OpCode::Equal, Value::String(a), Value::String(b)) => Ok(value!(a == b)),
                    (operation, left, right) => Err(EvaluationError::InvalidBinaryOp {
                        operation,
                        left,
                        right,
                    }),
                }
            }
            Expression::Transform {
                name,
                subject,
                args,
            } => {
                let subject = self.eval_ast(*subject, context)?;
                let mut args_arr = Vec::new();
                args_arr.push(subject);
                if let Some(args) = args {
                    for arg in args {
                        args_arr.push(self.eval_ast(*arg, context)?);
                    }
                }
                let f = self
                    .transforms
                    .get(&name)
                    .ok_or(EvaluationError::UnknownTransform(name))?;
                f(&args_arr).map_err(|e| e.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json as value;

    #[test]
    fn test_literal() {
        assert_eq!(Evaluator::new().eval("1").unwrap(), value!(1.0));
    }

    #[test]
    fn test_binary_expression_addition() {
        assert_eq!(Evaluator::new().eval("1 + 2").unwrap(), value!(3.0));
    }

    #[test]
    fn test_binary_expression_multiplication() {
        assert_eq!(Evaluator::new().eval("2 * 3").unwrap(), value!(6.0));
    }

    #[test]
    fn test_precedence() {
        assert_eq!(Evaluator::new().eval("2 + 3 * 4").unwrap(), value!(14.0));
    }

    #[test]
    fn test_parenthesis() {
        assert_eq!(Evaluator::new().eval("(2 + 3) * 4").unwrap(), value!(20.0));
    }

    #[test]
    fn test_string_concat() {
        assert_eq!(
            Evaluator::new().eval("'Hello ' + 'World'").unwrap(),
            value!("Hello World")
        );
    }

    #[test]
    fn test_true_comparison() {
        assert_eq!(Evaluator::new().eval("2 > 1").unwrap(), value!(true));
    }

    #[test]
    fn test_false_comparison() {
        assert_eq!(Evaluator::new().eval("2 <= 1").unwrap(), value!(false));
    }

    #[test]
    fn test_boolean_logic() {
        assert_eq!(
            Evaluator::new()
                .eval("'foo' && 6 >= 6 && 0 + 1 && true")
                .unwrap(),
            value!(true)
        );
    }

    #[test]
    fn test_identifier() {
        let context = value!({"a": 1.0});
        assert_eq!(
            Evaluator::new().eval_in_context("a", context).unwrap(),
            value!(1.0)
        );
    }

    #[test]
    fn test_identifier_chain() {
        let context = value!({"a": {"b": 2.0}});
        assert_eq!(
            Evaluator::new().eval_in_context("a.b", context).unwrap(),
            value!(2.0)
        );
    }

    #[test]
    #[should_panic]
    fn test_context_filter_arrays() {
        let context = value!({
            "foo": {
                "bar": [
                    {"tek": "hello"},
                    {"tek": "baz"},
                    {"tok": "baz"},
                ]
            }
        });
        assert_eq!(
            Evaluator::new()
                .eval_in_context("foo.bar[.tek == 'baz']", &context)
                .unwrap(),
            value!([{"tek": "baz"}])
        );
    }

    #[test]
    fn test_context_array_index() {
        let context = value!({
            "foo": {
                "bar": [
                    {"tek": "hello"},
                    {"tek": "baz"},
                    {"tok": "baz"},
                ]
            }
        });
        assert_eq!(
            Evaluator::new()
                .eval_in_context("foo.bar[1].tek", context)
                .unwrap(),
            value!("baz")
        );
    }

    #[test]
    #[should_panic]
    fn test_object_expression_properties() {
        let context = value!({"foo": {"baz": {"bar": "tek"}}});
        assert_eq!(
            Evaluator::new()
                .eval_in_context("foo['ba' + 'z']", &context)
                .unwrap(),
            value!("tek")
        );
    }

    #[test]
    fn test_divfloor() {
        assert_eq!(Evaluator::new().eval("7 // 2").unwrap(), value!(3.0));
    }

    #[test]
    fn test_empty_object_literal() {
        assert_eq!(Evaluator::new().eval("{}").unwrap(), value!({}));
    }

    #[test]
    fn test_object_literal_strings() {
        assert_eq!(
            Evaluator::new().eval("{'foo': {'bar': 'tek'}}").unwrap(),
            value!({"foo": {"bar": "tek"}})
        );
    }

    #[test]
    #[should_panic]
    fn test_object_literal_identifiers() {
        assert_eq!(
            Evaluator::new().eval("{foo: {bar: 'tek'}}").unwrap(),
            value!({"foo": {"bar": "tek"}})
        );
    }

    #[test]
    #[should_panic]
    fn test_object_literal_properties() {
        assert_eq!(
            Evaluator::new().eval("{foo: 'bar'}.foo").unwrap(),
            value!("bar")
        );
    }

    #[test]
    fn test_array_literal() {
        assert_eq!(
            Evaluator::new().eval("['foo', 1+2]").unwrap(),
            value!(["foo", 3.0])
        );
    }

    #[test]
    #[should_panic]
    fn test_array_literal_indexing() {
        assert_eq!(Evaluator::new().eval("[1, 2, 3][1]").unwrap(), value!(2.0));
    }

    #[test]
    fn test_in_operator_string() {
        assert_eq!(
            Evaluator::new().eval("'bar' in 'foobartek'").unwrap(),
            value!(true)
        );
        assert_eq!(
            Evaluator::new().eval("'baz' in 'foobartek'").unwrap(),
            value!(false)
        );
    }

    #[test]
    fn test_in_operator_array() {
        assert_eq!(
            Evaluator::new()
                .eval("'bar' in ['foo', 'bar', 'tek']")
                .unwrap(),
            value!(true)
        );
        assert_eq!(
            Evaluator::new()
                .eval("'baz' in ['foo', 'bar', 'tek']")
                .unwrap(),
            value!(false)
        );
    }

    #[test]
    #[should_panic]
    fn test_conditional_expression() {
        assert_eq!(Evaluator::new().eval("'foo' ? 1 : 2").unwrap(), value!(1));
        assert_eq!(Evaluator::new().eval("'' ? 1 : 2").unwrap(), value!(2));
    }

    #[test]
    fn test_arbitrary_whitespace() {
        assert_eq!(
            Evaluator::new().eval("(\t2\n+\n3) *\n4\n\r\n").unwrap(),
            value!(20.0)
        );
    }

    #[test]
    fn test_non_integer() {
        assert_eq!(Evaluator::new().eval("1.5 * 3.0").unwrap(), value!(4.5));
    }

    #[test]
    fn test_string_literal() {
        assert_eq!(
            Evaluator::new().eval("'hello world'").unwrap(),
            value!("hello world")
        );
        assert_eq!(
            Evaluator::new().eval("\"hello world\"").unwrap(),
            value!("hello world")
        );
    }

    #[test]
    fn test_string_escapes() {
        assert_eq!(Evaluator::new().eval("'a\\'b'").unwrap(), value!("a'b"));
        assert_eq!(Evaluator::new().eval("\"a\\\"b\"").unwrap(), value!("a\"b"));
    }

    #[test]
    // Test a very simple transform that applies to_lowercase to a string
    fn test_simple_transform() {
        let evaluator = Evaluator::new().with_transform("lower", |v: &[Value]| {
            let s = v
                .get(0)
                .expect("There should be one argument!")
                .as_str()
                .expect("Should be a string!");
            Ok(value!(s.to_lowercase()))
        });
        assert_eq!(evaluator.eval("'T_T'|lower").unwrap(), value!("t_t"));
    }

    #[test]
    // Test returning an UnknownTransform error if a transform is unknown
    fn test_missing_transform() {
        let err = Evaluator::new().eval("'hello'|world").unwrap_err();
        if let EvaluationError::UnknownTransform(transform) = err {
            assert_eq!(transform, "world")
        } else {
            panic!("Should have thrown an unknown transform error")
        }
    }

    #[test]
    fn test_add_multiple_transforms() {
        let evaluator = Evaluator::new()
            .with_transform("sqrt", |v: &[Value]| {
                let num = v
                    .first()
                    .expect("There should be one argument!")
                    .as_f64()
                    .expect("Should be a valid number!");
                Ok(value!(num.sqrt() as u64))
            })
            .with_transform("square", |v: &[Value]| {
                let num = v
                    .first()
                    .expect("There should be one argument!")
                    .as_f64()
                    .expect("Should be a valid number!");
                Ok(value!((num as u64).pow(2)))
            });

        assert_eq!(evaluator.eval("4|square").unwrap(), value!(16));
        assert_eq!(evaluator.eval("4|sqrt").unwrap(), value!(2));
        assert_eq!(evaluator.eval("4|square|sqrt").unwrap(), value!(4));
    }

    #[test]
    fn test_transform_with_argument() {
        let evaluator = Evaluator::new().with_transform("split", |args: &[Value]| {
            let s = args
                .first()
                .expect("Should be a first argument!")
                .as_str()
                .expect("Should be a string!");
            let c = args
                .get(1)
                .expect("There should be a second argument!")
                .as_str()
                .expect("Should be a string");
            let res: Vec<&str> = s.split_terminator(c).collect();
            Ok(value!(res))
        });

        assert_eq!(
            evaluator.eval("'John Doe'|split(' ')").unwrap(),
            value!(vec!["John", "Doe"])
        );
    }

    #[derive(Debug, thiserror::Error)]
    enum CustomError {
        #[error("Invalid argument in transform!")]
        InvalidArgument,
    }

    #[test]
    fn test_custom_error_message() {
        let evaluator = Evaluator::new().with_transform("error", |_: &[Value]| {
            Err(CustomError::InvalidArgument.into())
        });
        let res = evaluator.eval("1234|error");
        assert!(res.is_err());
        if let EvaluationError::CustomError(e) = res.unwrap_err() {
            assert_eq!(e.to_string(), "Invalid argument in transform!")
        } else {
            panic!("Should have returned a Custom error!")
        }
    }
}
