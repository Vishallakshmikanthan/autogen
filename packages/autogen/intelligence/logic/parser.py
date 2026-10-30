"""Recursive descent LL(1) parser and lexer for propositional logic formulas (SYS-07, Module M08)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Sequence

from autogen.intelligence.logic.ast import (
    And,
    ConstantBool,
    Equiv,
    Formula,
    Implies,
    Not,
    Or,
    PropVar,
)


class TokenType(str, Enum):
    VAR = "VAR"
    TRUE = "TRUE"
    FALSE = "FALSE"
    NOT = "NOT"          # ~ or !
    AND = "AND"          # & or && or and
    OR = "OR"            # | or || or or
    IMPLIES = "IMPLIES"  # => or ->
    EQUIV = "EQUIV"      # <=> or <->
    LPAREN = "LPAREN"    # (
    RPAREN = "RPAREN"    # )
    EOF = "EOF"


@dataclass(frozen=True, slots=True)
class Token:
    token_type: TokenType
    value: str
    position: int


TOKEN_SPECIFICATION = [
    (TokenType.EQUIV, r"<=>|<->"),
    (TokenType.IMPLIES, r"=>|->"),
    (TokenType.AND, r"&&|&|\band\b"),
    (TokenType.OR, r"\|\||\||\bor\b"),
    (TokenType.NOT, r"~|!|\bnot\b"),
    (TokenType.LPAREN, r"\("),
    (TokenType.RPAREN, r"\)"),
    (TokenType.TRUE, r"\bTrue\b|\bTRUE\b"),
    (TokenType.FALSE, r"\bFalse\b|\bFALSE\b"),
    (TokenType.VAR, r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("SKIP", r"[ \t\r\n]+"),
]

TOKEN_REGEX = re.compile(
    "|".join(f"(?P<{t.name if isinstance(t, TokenType) else t}>{pattern})" for t, pattern in TOKEN_SPECIFICATION)
)


def tokenize(text: str) -> list[Token]:
    """Scans and converts input text into a sequence of lexical tokens."""
    tokens: list[Token] = []
    for match in TOKEN_REGEX.finditer(text):
        kind = match.lastgroup
        value = match.group()
        if kind == "SKIP":
            continue
        if kind is None:
            raise SyntaxError(f"Unexpected token at position {match.start()}: {value}")
        tokens.append(Token(TokenType[kind], value, match.start()))
    tokens.append(Token(TokenType.EOF, "", len(text)))
    return tokens


class PropositionalParser:
    """Recursive descent parser translating tokens into Formula AST nodes."""

    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = tokens
        self.cursor = 0

    def parse(self) -> Formula:
        expr = self._parse_equiv()
        if not self._check(TokenType.EOF):
            tok = self._peek()
            raise SyntaxError(f"Unexpected token '{tok.value}' at position {tok.position}. Expected end of formula.")
        return expr

    def _parse_equiv(self) -> Formula:
        expr = self._parse_implies()
        while self._match(TokenType.EQUIV):
            right = self._parse_implies()
            expr = Equiv(expr, right)
        return expr

    def _parse_implies(self) -> Formula:
        expr = self._parse_or()
        if self._match(TokenType.IMPLIES):
            # Implication is right-associative: A => B => C is A => (B => C)
            consequent = self._parse_implies()
            expr = Implies(expr, consequent)
        return expr

    def _parse_or(self) -> Formula:
        operands = [self._parse_and()]
        while self._match(TokenType.OR):
            operands.append(self._parse_and())
        if len(operands) == 1:
            return operands[0]
        return Or(*operands)

    def _parse_and(self) -> Formula:
        operands = [self._parse_not()]
        while self._match(TokenType.AND):
            operands.append(self._parse_not())
        if len(operands) == 1:
            return operands[0]
        return And(*operands)

    def _parse_not(self) -> Formula:
        if self._match(TokenType.NOT):
            operand = self._parse_not()
            return Not(operand)
        return self._parse_primary()

    def _parse_primary(self) -> Formula:
        tok = self._peek()
        if self._match(TokenType.TRUE):
            return ConstantBool(True)
        if self._match(TokenType.FALSE):
            return ConstantBool(False)
        if self._match(TokenType.VAR):
            return PropVar(tok.value)
        if self._match(TokenType.LPAREN):
            inner = self._parse_equiv()
            self._consume(TokenType.RPAREN, "Expected closing ')' after expression.")
            return inner
        raise SyntaxError(f"Unexpected token '{tok.value}' at position {tok.position}. Expected variable or '('.")

    def _peek(self) -> Token:
        return self.tokens[self.cursor]

    def _check(self, token_type: TokenType) -> bool:
        return self._peek().token_type == token_type

    def _match(self, token_type: TokenType) -> bool:
        if self._check(token_type):
            self.cursor += 1
            return True
        return False

    def _consume(self, token_type: TokenType, error_msg: str) -> Token:
        if self._check(token_type):
            tok = self._peek()
            self.cursor += 1
            return tok
        curr = self._peek()
        raise SyntaxError(f"{error_msg} (got '{curr.value}' at position {curr.position})")


def parse_formula(text: str) -> Formula:
    """Parses a symbolic string representation into a Formula AST."""
    tokens = tokenize(text)
    parser = PropositionalParser(tokens)
    return parser.parse()
