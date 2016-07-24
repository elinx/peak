import sys
from lang_lex import *

characters = 'a = 2; b = 3'

if __name__ == '__main__':
    tokens = lang_lex(characters)
    for token in tokens:
        print(token)