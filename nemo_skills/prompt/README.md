This folder contains a collection of prompts used in our codebase. The majority of the prompts can be found
inside "generic" folder and are meant to be used by default.

In some cases we also have model-specific prompts if they were identified to work much better for a specific
model family or are required to reproduced published evals (e.g. see llama3 folder).

Finally, there is a "judge" folder which contains prompts for various LLM-as-a-judge use-cases.

TODO: add docs on how prompt is constructed (for now see utils:Prompt class)