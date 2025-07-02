import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import xml.etree.ElementTree as ET
import pandas as pd

# Código modificado de
# https://huggingface.co/dejanseo/Query-Quality-Classifier/blob/main/app/app.py

# 2021, 2022, clef
corpus = "clef"


def fetch_topics(corpus):
    tree = ET.parse(f"./topics/topics_{corpus}.xml")
    root = tree.getroot()
    topics_xml = root.findall(
        'query') if corpus == "clef" else root.findall('topic')

    topics = {}
    for topic in topics_xml:
        if corpus == "clef":
            id = topic.find('id').text
            if id[-1] not in ["4", "5", "6"]: continue
            topics[topic.find('id').text] = {
                "number": topic.find('id').text,
                # Change with respect to the previous programs
                "title": topic.find('title').text,
                "narrative": topic.find('narrative').text if topic.find('narrative') is not None else ""
            }
        elif corpus == "2022":
            topics[topic.find('number').text] = {
                "number": topic.find('number').text,
                "title": topic.find('query').text,
                "description": topic.find('question').text,
                "disclaimer": topic.find('disclaimer').text,
                "answer": topic.find('answer').text,
                "evidence": topic.find('evidence').text,
                "narrative": topic.find('background').text
            }
        elif corpus == "2021":
            topics[topic.find('number').text] = {
                "number": topic.find('number').text,
                "title": topic.find('query').text,
                "description": topic.find('description').text,
                "disclaimer": topic.find('disclaimer').text,
                "answer": topic.find('stance').text,
                "evidence": topic.find('evidence').text,
                "narrative": topic.find('narrative').text
            }
        else:
            topics[topic.find('number').text] = {
                "number": topic.find('number').text,
                "title": topic.find('title').text,
                "description": topic.find('description').text,
                "answer": topic.find('answer').text,
                "evidence": topic.find('evidence').text,
                "narrative": topic.find('narrative').text
            }
    return topics

def classify_query(query):
    inputs = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        softmax_scores = torch.softmax(logits, dim=1).cpu().numpy()[0]
        confidence = softmax_scores[1] * 100  # Confidence for well-formed class
    
    return confidence


def main():

    topics = fetch_topics(corpus)
    
    confidences = {}
    for topic_id, topic in topics.items():
        query = topic["title"]

        confidence = classify_query(query)

        if confidence >= 50:
            print(f"Query Score: {confidence:.2f}% Most likely doesn't require query expansion.")
        else:
            print(f"The query is likely not well-formed with a score of {100 - confidence:.2f}% and most likely requires query expansion.")

        confidences[topic_id] = confidence
    
    s = pd.Series(confidences, index=list(confidences.keys()), name="confidence")
    s.to_csv(f"./confidences_query_quality_classifier/confidences_{corpus}.csv",  index_label="topic")


if __name__ == "__main__":
    # Load the model and tokenizer from the Hugging Face Model Hub
    model_name = 'dejanseo/Query-Quality-Classifier'
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForSequenceClassification.from_pretrained(model_name)
    # Set the model to evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    main()