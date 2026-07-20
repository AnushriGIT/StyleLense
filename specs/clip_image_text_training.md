# Specification: CLIP Image-Text Fine-Tuning (clipImage&Text)

This specification defines the behavior of the data processing and contrastive training logic for the text-and-image CLIP model fine-tuning using `shop_captions.json`.

## Feature: Dataset Construction
We build a `CLIPShopCaptionDataset` that loads shop image paths and pairs them with captions retrieved from `shop_captions.json` using the item's unique `item_id`.

### Scenario: Pairing shop image with caption
  Given an item with `item_id` "id_00000001" and `shop_image` "img/SHOP/id_00000001/shop_01.jpg"
  And `shop_captions.json` maps "id_00000001" to "a polyester t-shirt"
  When `CLIPShopCaptionDataset.__getitem__` processes this item
  Then it should return a PIL Image of "img/SHOP/id_00000001/shop_01.jpg"
  And the caption "a polyester t-shirt"

### Scenario: Handling missing captions gracefully
  Given an item with `item_id` "id_99999999"
  And `shop_captions.json` does not contain "id_99999999"
  When `CLIPShopCaptionDataset.__getitem__` processes this item
  Then it should return the caption "a piece of clothing" (or an empty string fallback)

---

## Feature: Symmetric InfoNCE Loss
To align image and text embeddings, we apply symmetric Contrastive Language-Image Pretraining (InfoNCE) loss.

### Scenario: Symmetric loss calculation
  Given a batch of image embeddings `I` and text embeddings `T`
  When computing the loss
  Then the logits should be computed as `(I @ T.T) * logit_scale`
  And the text-to-image loss should be `CrossEntropyLoss(logits, targets)`
  And the image-to-text loss should be `CrossEntropyLoss(logits.T, targets)`
  And the total loss should be `(text_to_image_loss + image_to_text_loss) / 2.0`

---

## Feature: Freezing Strategy
To preserve general features and prevent catastrophic forgetting, the early layers of CLIP should remain frozen.

### Scenario: Partial model freezing
  Given the CLIP model
  When initialized for training
  Then the Vision Transformer parameters (except the projection layer or top layers) should have `requires_grad = False`
  And the Text Transformer parameters (except the projection layer or top layers) should have `requires_grad = False`
