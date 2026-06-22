# Drafter
## Property
Class attributes
    name : str
    model_free : bool

## Method
### Public
```
1. get_draft_token # Abstract

Input : 
    prompt_token : List[]
    n_token : int
    user_id : Optional[str]
    hidden_state : Optional[str]

Return : 
    draft_token : List[List[]]

2. build_datastore # Abstract

Input : 
    token : List[]

Return : 
    status : bool

3. describe_datastore # Abstract

Return :
    peak_data : List[]
    total_data : int
```