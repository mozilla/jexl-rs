#[derive(Debug, PartialEq)]
pub enum Expression {
    Number(f64),
    String(String),
    Boolean(bool),
    Array(Vec<Box<Expression>>),
    Object(Vec<(String, Box<Expression>)>),
    IdentifierSequence(Vec<Box<Expression>>),
    BinaryOperation {
        operation: OpCode,
        left: Box<Expression>,
        right: Box<Expression>,
    },
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum OpCode {
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    And,
    Or,
    Modulus,
    Exponent,
    In,
}
