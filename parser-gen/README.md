# jexl-parser generator

We do not use `lalrpop` to generate the parser during the `jexl-parser` build phase.
Why? because doing so introduces a lot of dependencies in Firefox's build. So instead we check-in the generated parser file, and run a CI job to ensure it's always fresh.
