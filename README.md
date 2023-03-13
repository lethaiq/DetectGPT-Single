# DetectGPT-Single
DetectGPT code but for inferencing on a single document.
All credits go to **https://github.com/eric-mitchell/detect-gpt**

## Input
```
text = """Maryland's environmental protection agency is suing a Prince George's County recycling outfit, alleging that the company has violated anti-pollution laws for years at two rat-infested, oil-leaking, garbage-strewn sites in Cheverly and Baltimore. The 71-page complaint, filed on behalf of the Maryland Department of the Environment in Prince George's County Circuit Court this month, lays out environmental violations since December 2014 at two properties controlled by World Recycling Company and its affiliates, Pride Rock and Small World Real Estate."""
```
```
python main.py --output_name main \
            --base_model_name gpt2-xl \
            --mask_filling_model_name t5-large \
            --n_perturbation_list 120 \
            --pct_words_masked 0.3 \
            --span_length 2 \
            --mask_top_p 0.9 \
            --text "$text"
```
## Output
```
perturbation_mode d predictions: {'real': [0.02245943347613011]}
perturbation_mode z predictions: {'real': [0.23339302061208503]}
```
