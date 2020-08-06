pub mod ast;
mod parser {
    include!(concat!(env!("OUT_DIR"), "/parser.rs"));
}

pub use lalrpop_util::ParseError;

pub use crate::parser::Token;

pub struct Parser {}

impl Parser {
    pub fn parse(input: &str) -> Result<ast::Expression, ParseError<usize, Token, &str>> {
        Ok(*parser::ExpressionParser::new().parse(input)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expression, OpCode};

    #[test]
    fn literal() {
        assert_eq!(Parser::parse("1"), Ok(Expression::Number(1.0)));
    }

    #[test]
    fn binary_expression() {
        assert_eq!(
            Parser::parse("1+2"),
            Ok(Expression::BinaryOperation {
                operation: OpCode::Add,
                left: Box::new(Expression::Number(1.0)),
                right: Box::new(Expression::Number(2.0)),
            }),
        );
    }

    #[test]
    fn binary_expression_whitespace() {
        assert_eq!(Parser::parse("1  +     2 "), Parser::parse("1+2"),);
    }
}
