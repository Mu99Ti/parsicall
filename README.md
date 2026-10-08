# ParsiCall

ParsiCall is an OpenAI-compatible proxy that repairs malformed tool calls from small local LLMs: it rescues structurally broken JSON, prose/XML-ish tool-call emissions, and truncated arguments, and it canonicalizes Persian argument values (Persian digits and Jalali dates, Arabic-vs-Persian characters, ZWNJ/space drift, toman/rial amounts, Persian enum values) into schema-valid forms — with an auditable record of every mutation — so agent turns never abort on one bad call and business data never silently corrupts.

**Status:** MVP in progress.
