# AES-IELTS
Automatic essay scoring for IELTS part 2 writing essay

## Dataset
Unfortunately there were no public dataset for IELTS part 2 writing essay. So I have to create our own dataset. 

I have parsed the IELTS writing essays from [IELTS-practice](https://www.ielts-practice.org/). The total number of parsed essays is 2664. Unfortunately, there are only good score essays in the Web, so dataset is highly biased to good score essays. Therefore, the best application will be differentiating between good scrores. 

## Model
Backbone model is DistilBertForSequenceClassification from [transformers](https://github.com/huggingface/transformers). 
I have chosen standard regression head with MSE loss.

Trained model is uploaded to [tkharisov7/aes-ielts](https://huggingface.co/tkharisov7/aes-ielts). Also evaluation results can bee seen there.

## Deployment
For deployment purposes Streamlit and Python3 is used. 

Hosting is managed by HuggingFace Spaces. **You can try it [here](https://huggingface.co/spaces/tkharisov7/aes-ielts-space)**