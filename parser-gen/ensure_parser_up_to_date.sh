#!/usr/bin/env bash

cargo run --bin parser-gen

CHANGES=$(git status --porcelain)

if [ -n "$CHANGES" ]; then
    git status
    echo -e "\033[0;31mThe parser file looks outdated (git status is dirty). You can fix this yourself by running cargo run --bin parser-gen and then committing the result"
fi
