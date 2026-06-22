# Verifier
## Method
### Public
```
1. verify_with_history # Abstract

Input : 
    prompt_token : List[]
    draft_token : List[List[]]

Return : 
    draft_token : List[List[]]

2. verify_with_llm # Abstract

Input : 
    token : List[]

Return : 
    status : bool
```