use jexl_parser::{
    ast::{Expression, OpCode},
    ParseError, Parser, Token,
};
use serde_json::{json as value, Value};

const EPSILON: f64 = 0.000001f64;

#[derive(Debug, PartialEq)]
pub enum EvaluationError<'a> {
    ParseError(Box<ParseError<usize, Token<'a>, &'a str>>),
    InvalidBinaryOp {
        left: Value,
        right: Value,
        operation: OpCode,
    },
    UnknownTransform(String),
    DuplicateObjectKey(String),
    InvalidContext,
    InvalidIndexType,
}

impl<'a> From<ParseError<usize, Token<'a>, &'a str>> for EvaluationError<'a> {
    fn from(cause: ParseError<usize, Token<'a>, &'a str>) -> Self {
        EvaluationError::ParseError(Box::new(cause))
    }
}

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
            Value::String(s) => s.len() > 0,
            // It would be better if these depended on the contents of the
            // object (empty array/object is falsey, non-empty is truthy, like
            // in Python) but this matches JS semantics. Is it worth changing?
            Value::Array(_) => true,
            Value::Object(_) => true,
        }
    }
}

type Context = Value;

pub struct Evaluator {}

impl Evaluator {
    pub fn new() -> Self {
        Self {}
    }

    pub fn eval<'a>(&self, input: &'a str) -> Result<Value, EvaluationError<'a>> {
        let context = value!({});
        self.eval_in_context(input, &context)
    }

    pub fn eval_in_context<'a, 'b>(
        &self,
        input: &'a str,
        context: impl Into<&'b Context>,
    ) -> Result<Value, EvaluationError<'a>> {
        let tree = Parser::parse(input)?;
        let context = context.into();
        if !context.is_object() {
            return Err(EvaluationError::InvalidContext);
        }
        self.eval_ast(tree, context)
    }

    fn eval_ast<'a>(
        &self,
        ast: Expression,
        context: &Context,
    ) -> Result<Value, EvaluationError<'a>> {
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

                    (operation, left, right) => Err(EvaluationError::InvalidBinaryOp {
                        operation,
                        left,
                        right,
                    }),
                }
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
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("1"), Ok(value!(1.0)));
    }

    #[test]
    fn test_binary_expression_addition() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("1 + 2"), Ok(value!(3.0)));
    }

    #[test]
    fn test_binary_expression_multiplication() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("2 * 3"), Ok(value!(6.0)));
    }

    #[test]
    fn test_precedence() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("2 + 3 * 4"), Ok(value!(14.0)));
    }

    #[test]
    fn test_parenthesis() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("(2 + 3) * 4"), Ok(value!(20.0)));
    }

    #[test]
    fn test_string_concat() {
        let evaluator = Evaluator::new();
        assert_eq!(
            evaluator.eval("'Hello ' + 'World'"),
            Ok(value!("Hello World"))
        );
    }

    #[test]
    fn test_true_comparison() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("2 > 1"), Ok(value!(true)));
    }

    #[test]
    fn test_false_comparison() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("2 <= 1"), Ok(value!(false)));
    }

    #[test]
    fn test_boolean_logic() {
        let evaluator = Evaluator::new();
        assert_eq!(
            evaluator.eval("'foo' && 6 >= 6 && 0 + 1 && true"),
            Ok(value!(true))
        );
    }

    #[test]
    fn test_identifier() {
        let context = value!({"a": 1.0});
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval_in_context("a", &context), Ok(value!(1.0)));
    }

    #[test]
    fn test_identifier_chain() {
        let context = value!({"a": {"b": 2.0}});
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval_in_context("a.b", &context), Ok(value!(2.0)));
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
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
        let evaluator = Evaluator::new();
        assert_eq!(
            evaluator.eval_in_context("foo.bar[.tek == 'baz']", &context),
            Ok(value!([{"tek": "baz"}]))
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
        let evaluator = Evaluator::new();
        assert_eq!(
            evaluator.eval_in_context("foo.bar[1].tek", &context),
            Ok(value!("baz"))
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_object_expression_properties() {
        let context = value!({"foo": {"baz": {"bar": "tek"}}});
        let evaluator = Evaluator::new();
        assert_eq!(
            evaluator.eval_in_context("foo['ba' + 'z']", &context),
            Ok(value!("tek"))
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_missing_transform_exception() {
        let evaluator = Evaluator::new();
        assert_eq!(
            evaluator.eval("'hello'|world"),
            Err(EvaluationError::UnknownTransform("world".to_string()))
        );
    }

    #[test]
    fn test_divfloor() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("7 // 2"), Ok(value!(3.0)));
    }

    #[test]
    fn test_empty_object_literal() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("{}"), Ok(value!({})));
    }

    #[test]
    fn test_object_literal_strings() {
        let evaluator = Evaluator::new();
        assert_eq!(
            evaluator.eval("{'foo': {'bar': 'tek'}}"),
            Ok(value!({"foo": {"bar": "tek"}}))
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_object_literal_identifiers() {
        let evaluator = Evaluator::new();
        assert_eq!(
            evaluator.eval("{foo: {bar: 'tek'}}"),
            Ok(value!({"foo": {"bar": "tek"}}))
        );
    }

    /*
    // TODO needs transforms

    def test_transforms():
        config = JEXLConfig(
            {'half': lambda x: x / 2},
            default_binary_operators,
            default_unary_operators
        )
        evaluator = Evaluator(config)
        result = evaluator.evaluate(tree('foo|half + 3'), {'foo': 10})
        assert result == 8

    def test_transforms_multiple_arguments():
        config = JEXLConfig(
            binary_operators=default_binary_operators,
            unary_operators=default_unary_operators,
            transforms={
                'concat': lambda val, a1, a2, a3: val + ': ' + a1 + a2 + a3,
            }
        )
        evaluator = Evaluator(config)
        result = evaluator.evaluate(tree('"foo"|concat("baz", "bar", "tek")'))
        assert result == 'foo: bazbartek'

    */

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_object_literal_properties() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("{foo: 'bar'}.foo"), Ok(value!("bar")));
    }

    #[test]
    fn test_array_literal() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("['foo', 1+2]"), Ok(value!(["foo", 3.0])));
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_array_literal_indexing() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("[1, 2, 3][1]"), Ok(value!(2.0)));
    }

    #[test]
    fn test_in_operator_string() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("'bar' in 'foobartek'"), Ok(value!(true)));
        assert_eq!(evaluator.eval("'baz' in 'foobartek'"), Ok(value!(false)));
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_in_operator_array() {
        let evaluator = Evaluator::new();
        assert_eq!(
            evaluator.eval("'bar' in ['foo', 'bar', 'tek']"),
            Ok(value!(true))
        );
        assert_eq!(
            evaluator.eval("'baz' in ['foo', 'bar', 'tek']"),
            Ok(value!(false))
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_conditional_expression() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("'foo' ? 1 : 2"), Ok(value!(1)));
        assert_eq!(evaluator.eval("'' ? 1 : 2"), Ok(value!(2)));
    }

    #[test]
    fn test_arbitrary_whitespace() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("(\t2\n+\n3) *\n4\n\r\n"), Ok(value!(20.0)));
    }

    #[test]
    fn test_non_integer() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("1.5 * 3.0"), Ok(value!(4.5)))
    }

    #[test]
    fn test_string_literal() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("'hello world'"), Ok(value!("hello world")));
        assert_eq!(evaluator.eval("\"hello world\""), Ok(value!("hello world")));
    }

    #[test]
    fn test_string_escapes() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.eval("'a\\'b'"), Ok(value!("a'b")));
        assert_eq!(evaluator.eval("\"a\\\"b\""), Ok(value!("a\"b")));
    }
}
