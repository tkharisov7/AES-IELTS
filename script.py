import streamlit as st

st.markdown("# Automatic Essay Scoring for IELTS Writing Task 2")
st.markdown("## Please enter your question and essay below:")
st.markdown("**Disclaimer: This is a demo app and the results are not accurate. Model is trained on small dataset and is not robust enough to generalize well. Main application is to determine scores from 6 to 9. Scores below 6 are not accurate.**")

st.markdown("### Question:")
question = st.text_input("Enter your question here")

st.markdown("### Essay:")
essay = st.text_input("Enter your essay here")

@st.cache_resource
def get_pipeline():
    from transformers import Pipeline

    class AESIELTSPipeline(Pipeline):
        def _sanitize_parameters(self, **kwargs):
            return kwargs, {}, {}

        def preprocess(self, inputs):
            question, essay = inputs
            encoding = self.tokenizer(question, essay, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        def _forward(self, input):
            output = self.model(**input)
            return output[0].item()
        
        def postprocess(self, output):
            return output
        
    from transformers.pipelines import PIPELINE_REGISTRY
    from transformers import DistilBertForSequenceClassification

    PIPELINE_REGISTRY.register_pipeline(
        "aes-ielts",
        AESIELTSPipeline,
        pt_model=DistilBertForSequenceClassification
    )

    from transformers import pipeline
    pipe = pipeline("aes-ielts", model="tkharisov7/aes-ielts")
    return pipe

pipe = get_pipeline()
predictions = pipe((question, essay))

st.markdown("### Estimated Score:")

st.markdown(f"**{predictions}**")