# Specification: Local VLM Captioning (Moondream2)

This specification defines the behavior of the local VLM-based image caption generator and its post-processing mapping logic.

## Feature: Prompt Assembly with Attribute Context
To combine the existing ground-truth whitelisted annotations with the local generative VLM, we construct a prompt that includes the attribute terms.

### Scenario: Prompt construction with attributes
  Given the whitelisted attributes:
    | attributes |
    | cotton, striped, casual |
  When the prompt builder is executed
  Then it should generate a prompt template incorporating these words:
    "This garment is characterized by: cotton, striped, casual. Describe this item in a single descriptive sentence, focusing on its visual color, pattern style, material texture, and cut."

---

## Feature: Metadata Serialization & Structure
The generator script executes local batch prediction and writes output to `shop_captions.json`.

### Scenario: Creating valid metadata format
  Given a set of generated captions:
    | item_id | caption |
    | 00000001 | a blue cotton striped t-shirt with regular fit |
  When serialized to `shop_captions.json`
  Then it must match a valid JSON format of:
    ```json
    {
      "00000001": "a blue cotton striped t-shirt with regular fit"
    }
    ```
  And all captions must be single-line strings.
