# Prompts

## Case study on initial persona set generation
Take OnlineRetail as an example
### Instruction
<div style="background-color: #FFF9C4; padding: 15px; border-radius: 8px; border: 1px solid #FBC02D;">

**System Prompt:**  
Now you are an intelligent e-commerce domain assistant. You are skilled at summarizing, and capable of assigning high-level consumer personas based on a user's purchase behavior.

**User Prompt:**  
Take a deep breath and work according to the instructions step by step. 
    
Your goal is to identify users' shopping behaviors based on products they have bought and label them with a given set of personas. You need to select at least one persona, at most 5 personas from our given persona list. But make sure that for each assignment you should find strong evidence in their purchase transactions. Please keep the procedure as accurate as possible.

Please provide the output in json format. Prefer to return arrays instead of comma separated strings. The following is an explanation of your return format:

```json
{"user_number": ["Persona1", "Persona2", "Persona3"]}
```
And here is a specific example:
```json
{"12346": [ "Vegan/Vegetarian", "High-Protein Shopper", "Pet Owner"]}
```

In the case that you feel there does not exist any suitable persona from the given list that can properly describe a user's purchasing behavior, you can label the user as an 'unrepresentable' user as the following example:
```json
{"12999": ["Unrepresentable"]}
```
</div>

### Input Prompt
<div style="background-color: #F5F5F5; padding: 15px; border-radius: 8px; border: 1px solid #B0BEC5;">

Here is the persona list you should choose from: ***[PERSONA LIST]***

Remember that the user number (i.e., “user\_number” in the example) should be exactly from the given transaction data, do not make it wrong since it is crucial.

Here is the data of user 12358's transaction data for you to analyze: 

The user 12358 has totally purchased 13 unique products, we show each product name followed by its purchased times: he bought: FAIRY CAKE DESIGN UMBRELLA, 4 times; CERAMIC STRAWBERRY DESIGN MUG, 24 times; CERAMIC CAKE STAND + HANGING CAKES, 2 times; CERAMIC CAKE DESIGN SPOTTED PLATE, 12 times; DOORMAT FAIRY CAKE, 2 times; EDWARDIAN PARASOL PINK, 12 times; EDWARDIAN PARASOL NATURAL, 24 times; EDWARDIAN PARASOL RED, 24 times; EDWARDIAN PARASOL BLACK, 24 times; STRAWBERRY CERAMIC TRINKET BOX, 12 times; CERAMIC BOWL WITH STRAWBERRY DESIGN, 6 times; POSTAGE, 4 times; CERAMIC STRAWBERRY CAKE MONEY BANK, 36 times. Remind one more time that you can only select from the given 20 personas' list and only use the exactly given persona, you cannot use other words to describe. You do not need to explain how you get the result, so please respond no more than the required format.
</div>

### Generated Result
<div style="background-color: #E3F2FD; padding: 15px; border-radius: 8px; border: 1px solid #64B5F6;">

```json
{"12358": ["Home Decor Aficionado", "Vintage and Retro Enthusiast"]}
```
</div>

## Case study on user persona generation
Take user 12358 in OnlineRetail as an example
### Instruction
<div style="background-color: #FFF9C4; padding: 15px; border-radius: 8px; border: 1px solid #FBC02D;">

**System Prompt:**  
You are an assistant skilled at summarizing, capable of deducing high-level consumer keywords based on a user's purchases.

**User Prompt:**  
Take a deep breath and work according to the instructions step by step. 
    
Your goal is to identify users' shopping behaviors based on products they have bought and label them with a given set of personas. You need to select at least one persona, at most 5 personas from our given persona list. But make sure that for each assignment you should find strong evidence in their purchase transactions. Please keep the procedure as accurate as possible.

Please provide the output in json format. Prefer to return arrays instead of comma separated strings. The following is an explanation of your return format:

```json
{"user_number": ["Persona1", "Persona2", "Persona3"]}
```
And here is a specific example:
```json
{"12346": [ "Vegan/Vegetarian", "High-Protein Shopper", "Pet Owner"]}
```

In the case that you feel there does not exist any suitable persona from the given list that can properly describe a user's purchasing behavior, you can label the user as an 'unrepresentable' user as the following example:
```json
{"12999": ["Unrepresentable"]}
```
</div>

### Input Prompt
<div style="background-color: #F5F5F5; padding: 15px; border-radius: 8px; border: 1px solid #B0BEC5;">

Here is the persona list you should choose from: ***[PERSONA LIST]***

Remember that the user number (i.e., “user\_number” in the example) should be exactly from the given transaction data, do not make it wrong since it is crucial.

Here is the data of user 12358's transaction data for you to analyze: 

The user 12358 has totally purchased 13 unique products, we show each product name followed by its purchased times: he bought: FAIRY CAKE DESIGN UMBRELLA, 4 times; CERAMIC STRAWBERRY DESIGN MUG, 24 times; CERAMIC CAKE STAND + HANGING CAKES, 2 times; CERAMIC CAKE DESIGN SPOTTED PLATE, 12 times; DOORMAT FAIRY CAKE, 2 times; EDWARDIAN PARASOL PINK, 12 times; EDWARDIAN PARASOL NATURAL, 24 times; EDWARDIAN PARASOL RED, 24 times; EDWARDIAN PARASOL BLACK, 24 times; STRAWBERRY CERAMIC TRINKET BOX, 12 times; CERAMIC BOWL WITH STRAWBERRY DESIGN, 6 times; POSTAGE, 4 times; CERAMIC STRAWBERRY CAKE MONEY BANK, 36 times. Remind one more time that you can only select from the given 20 personas' list and only use the exactly given persona, you cannot use other words to describe. You do not need to explain how you get the result, so please respond no more than the required format.
</div>

### Generated Result
<div style="background-color: #E3F2FD; padding: 15px; border-radius: 8px; border: 1px solid #64B5F6;">

```json
{"12358": ["Home Decor Aficionado", "Vintage and Retro Enthusiast"]}
```
</div>