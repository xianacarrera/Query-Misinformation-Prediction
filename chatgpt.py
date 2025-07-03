import os
from openai import OpenAI
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import json
import configparser
import numpy as np
import pandas as pd


def get_prompt_variants(description, role=False, narrative=None, chain_of_thought=1, n=5):
    prompt = ""
    if role:
        prompt += f"You are a search engineer trying to improve the relevance, correctness and credibility of search results for health-related queries. "

    prompt += f"Given a query, you must provide a list of {n} alternative queries that express the same information need as the original one, but that "
    prompt += f"are phrased in such a way that they are more likely to retrieve relevant, correct and credible documents.\n"

    prompt += f"Query\n"
    prompt += f"A person has typed [{description}] into a search engine.\n"
    if narrative:
        prompt += f"They were looking for: {narrative}\n"

    if chain_of_thought > 0:
        prompt += f"Instructions\n"
        prompt += f"Let's think step by step:\n"
        prompt += f"Consider the underlying intent of the search.\n"

    if chain_of_thought == 2:
        prompt += f"Measure how prone the original query is to retrieve useful documents.\n"
        prompt += f"Measure how prone the original query is to retrieve supportive documents for the correct treatment of the query's question.\n"
        prompt += f"Measure how prone the original query is to retrieve credible documents.\n"
        prompt += f"Consider the aspects above and the relative importance of each, and produce "
    else:    # chain_of_thought == 1 or chain_of_thought == 0
        prompt += f"Produce "

    prompt += "an array of variant queries without providing any reasoning. Example: "
    prompt += f"[\"query variant 1\", \"query variant 2\", ...]"

    return prompt


def write_narrative_from_examples(query):
    # https://trec-health-misinfo.github.io/docs/TREC-2021-Health-Misinformation-Track-Assessing-Guidelines_Version-2.pdf
    prompt = f"Given the query [{query}], write a narrative that describes its information need in more detail "
    prompt += f"and provides a specific explanation of what is considered to be very-useful or useful information for the query.\n"
    # Topic 105 (2021)
    prompt += "For example, if the query is [Should I apply ice to a burn?], a good narrative could be: "
    prompt += "'Many people commonly put on ice on burns in an attempt to stop the burning and pain. A very useful document would discuss the effectiveness of using ice to treat burns. "
    prompt += "A useful document would help a user decide if putting ice on burns is a recommended treatment by providing information on recommended treatments for burns and may not discuss "
    prompt += "ice as a treatment, or a useful document may discuss benefits or concerns for application of ice to skin.'\n"
    # Topic 107 (2021)
    prompt += "Another example: if the query is [Does yoga improve the management of asthma?], a good narrative could be: "
    prompt += "'Asthma is a condition that causes shortness of breath or chest pains due to narrowing airways. Yoga is an increasingly popular low-impact exercise --- claiming relaxation, strength "
    prompt += "improvements and overall health benefits. A very useful document would discuss the effectiveness of yoga in managing asthma symptoms. A useful document would help a user make a decision "
    prompt += "about use of yoga to help manage asthma by providing information on recommended treatments for asthma management, or medical benefits of practicing yoga, or both.'\n"
    prompt += "Write just the narrative using a similar format as in the examples above. Do not include any other information and do not repeat the query in your answer.\n"
    return prompt


def write_narrative_from_style_description(query):
    prompt = f"Given the query [{query}], write a description that:\n"
    prompt += "1. Identifies the key terms and concepts in the query.\n"
    prompt += "2. Provides brief definitions or explanations for each term.\n"
    prompt += "3. Rephrases the query to highlight its underlying concern or focus.\n"
    prompt += "4. Adds relevant context to connect the terms meaningfully, if needed.\n"
    prompt += "5. Maintains an explanatory and neutral tone, ensuring clarity and conciseness.\n"
    prompt += "Write just the description. Do not include any other information and do not repeat the query in your answer.\n"
    return prompt


def deprecated_write_narrative_from_style_description(query):
    prompt = f"Given the query [{query}], write a narrative that describes its information need in more detail. Use the standard TREC format for narratives, with the following style guidelines:\n"
    prompt += "* The narrative should have the following structure:\n"
    prompt += "    - Begin by describing a specific health claim or rumor, such as health remedies or conspiracy theories.\n"
    prompt += "    - Then, provide context, such as sources of misinformation or typical misconceptions.\n"
    prompt += "    - Finally, outline specific criteria for helpful documents (those providing truthful and safe instructions) and harmful documents (those that mislead or fail to clarify risks).\n"
    prompt += "* The voice should be:\n"
    prompt += "    - Objective and neutral, delivering information without bias or emotional language. Focus on a clear presentation of facts.\n"
    prompt += "    - Authoritative and factual, providing scientifically grounded statements, particularly when clarifying health misinformation.\n"
    prompt += "    - Instructional, offering guidance on what readers should consider reliable information versus misleading information.\n"
    prompt += "* The tone should be:\n"
    prompt += "    - Informative and cautious, in a way that prevents misinformation by carefully explaining what constitutes helpful versus harmful information.\n"
    prompt += "    - Calm and reassuring, addressing potentially anxiety-inducing topics in a composed manner to reduce panic or confusion.\n"
    prompt += "    - Clear-cut and decisive, distinguishing between helpful and harmful documents in a straightforward, definitive way to reduce ambiguity.\n"
    prompt += "* Use a language style that is:\n"
    prompt += "    - Plain and accessible, with simple language that makes the content understandable to a wide audience.\n"
    prompt += "    - Concise and direct. Each narrative should avoid unnecessary detail, focusing on the essentials of what is helpful or harmful.\n"
    prompt += "    - Predictable. It should follow a consistent pattern that helps readers quickly differentiate between reliable and unreliable information.\n"
    prompt += "Write a complete narrative for the query in a single paragraph. Do not include any other information and do not repeat the query in your answer.\n"
    return prompt


def write_narrative_basic_prompt(query):
    prompt = f"Given the query [{query}], write a narrative detailing the information need and describing the characteristics of helpful and "
    prompt += "harmful documents. Write one paragraph and do not repeat the query in your answer."
    return prompt


def write_narrative_from_TREC(query):
    prompt = f"Given the query [{query}], write a narrative detailing the information need and describing the characteristics of helpful and "
    prompt += "harmful documents using the standard TREC format for narratives. Write one paragraph and do not repeat the query in your answer."
    return prompt


def get_prompt_evaluation(description, role=False, narrative=None, chain_of_thought=1):
    prompt = ""
    if role:
        prompt += f"You are a query quality predictor evaluating the effectiveness of a user query and the potential harmfulness of documents retrieved for that query. "

    prompt += f"Given a query, you must provide a score on an integer scale of 0 to 2 with the following meanings:\n"
    prompt += f"2 = good query. The relevant documents that would appear as search results would mainly provide information that is credible and correct\n"
    prompt += f"1 = average query. Some relevant documents that would appear as search results could provide information that is correct but has low or moderate credibility\n"
    prompt += f"0 = bad query. Most relevant documents that would appear as search results could provide incorrect information\n"

    prompt += f"Query\n"
    prompt += f"A person has typed [{description}] into a search engine.\n"
    if narrative:
        prompt += f"They were looking for: {narrative}\n"

    if chain_of_thought > 0:
        prompt += f"Instructions\n"
        prompt += f"Let's think step by step:\n"
        prompt += f"Consider the underlying intent of the search.\n"

    if chain_of_thought == 2:
        # prompt += f"Measure how prone the original query is to retrieve useful documents (U).\n"
        prompt += f"Measure how prone the original query is to retrieve supportive documents for the correct treatment of the query's question (S).\n"
        prompt += f"Measure how prone the original query is to retrieve credible documents (C).\n"
        prompt += f"Consider the aspects above and the relative importance of each, and decide"
    else:
        prompt += f"Decide"
    prompt += f" on the final score (H).\n"

    prompt += f"Produce a JSON score without providing any reasoning. Example: "
    if chain_of_thought == 2:
        prompt += f"{{\"S\": 0, \"C\": 2, \"H\": 0}}"
    else:
        prompt += f"{{\"H\": 1}}"

    return prompt


def get_passage_writing_prompt(query, n=10):
    prompt = f"Write passages to answer the question [{query}]. Each passage should correspond to a different interpretation, meaning or answer to the question. Write as many as necessary to fully capture all possibilities.\n"
    prompt += "Separate the passages with the mark '||PAS||'. Do not include anything else in your answer."
    return prompt

def get_prompt_controversy(query, role=True, judges = 1, factors=True):
    prompt = ""
    if role:
        prompt += f"You are an expert in information retrieval and search engine bias. "

    prompt += f"Given a query, you must determine its level of controversy within the context of health-related information retrieval.\n"
    prompt += f"Consider factors such as ambiguity, polarization in search results, potential misinformation and conflicting information.\n"
    if judges == 1:
        prompt += f"Think step by step and provide a score on an integer scale of 1 (not controversial) to 5 (highly controversial) for the query [{query}].\n"
        if factors:
            prompt += f"Your answer should be a JSON array of scores for the individual factors and the total score at the end. Do not include any textual description. Example: [1, 5, 2, 2, 3]\n"
        else:
            prompt += f"Your answer should be a single integer representing the total score. Do not include any other information."
    else:
        prompt += f"{judges} different controversy raters were asked to evaluate the query. Each rater used their own independent judgement.\n"
        prompt += f"Think step by step and provide {judges} scores on an integer scale of 1 (not controversial) to 5 (highly controversial) for the query [{query}].\n"
        prompt += f"Your answer should be a JSON array of scores without providing any reasoning. Example: [1, 5, 2, 2, 3]"
    return prompt


def chat_with_gpt4(client, prompt, temp=0.7):
    try:
        # Create a chat completion
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=temp,     # originally 0.2
            frequency_penalty=0.0
        )

        response_text = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens

        return {"response": response_text, "tokens_used": tokens_used}
    except Exception as e:
        return f"An error occurred: {str(e)}"


def fetch_topics(path='../TREC_2020_BEIR/original-misinfo-resources-2020/topics/misinfo-2020-topics.xml', corpus="2020"):
    tree = ET.parse(path)
    root = tree.getroot()
    topics_xml = root.findall(
        'query') if corpus == "clef" else root.findall('topic')

    topics = {}
    for topic in topics_xml:
        if corpus == "clef":
            topics[topic.find('id').text] = {
                "number": topic.find('id').text,
                "description": topic.find('title').text,
                "narrative": topic.find('narrative').text if topic.find('narrative') is not None else ""
            }
        elif corpus == "2022":
            topics[topic.find('number').text] = {
                "number": topic.find('number').text,
                "title": "" if topic.find('query') is None else topic.find('query').text,
                "description": topic.find('question').text,
                "disclaimer": topic.find('disclaimer').text,
                "answer": topic.find('answer').text,
                "evidence": topic.find('evidence').text,
                "narrative": topic.find('background').text
            }
        elif corpus == "2021":
            topics[topic.find('number').text] = {
                "number": topic.find('number').text,
                "title": "" if topic.find('query') is None else topic.find('query').text,
                "description": topic.find('description').text,
                "disclaimer": topic.find('disclaimer').text,
                "answer": topic.find('stance').text,
                "evidence": topic.find('evidence').text,
                "narrative": topic.find('narrative').text
            }
        else:
            topics[topic.find('number').text] = {
                "number": topic.find('number').text,
                "title": "" if topic.find('title') is None else topic.find('title').text,
                "description": topic.find('description').text,
                "answer": topic.find('answer').text,
                "evidence": topic.find('evidence').text,
                "narrative": topic.find('narrative').text
            }
    return topics


def save_xml(topics, variants, filename, n):
    # Save the variants to an xml file
    for i in range(1, n + 1):
        with open(f'{filename}_{i}.xml', 'w') as f:
            f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            f.write("<queries>\n" if corpus == "clef" else "<topics>\n")
            for topic_id in topics:
                if corpus == "2020":
                    f.write(f"\t<topic>\n")
                    f.write(f"\t\t<number>{topic_id}</number>\n")
                    f.write(
                        f"\t\t<title>{topics[topic_id]['title']}</title>\n")
                    f.write(
                        f"\t\t<description>{variants[topic_id][i-1]}</description>\n")
                    f.write(
                        f"\t\t<answer>{topics[topic_id]['answer']}</answer>\n")
                    f.write(
                        f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                    f.write(
                        f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                    f.write(f"\t</topic>\n")

                elif corpus == "2022":
                    f.write(f"\t<topic>\n")
                    f.write(f"\t\t<number>{topic_id}</number>\n")
                    f.write(
                        f"\t\t<question>{variants[topic_id][i-1]}</question>\n")
                    f.write(
                        f"\t\t<query>{topics[topic_id]['title']}</query>\n")
                    f.write(
                        f"\t\t<background>{topics[topic_id]['narrative']}</background>\n")
                    f.write(
                        f"\t\t<disclaimer>{topics[topic_id]['disclaimer']}</disclaimer>\n")
                    f.write(
                        f"\t\t<answer>{topics[topic_id]['answer']}</answer>\n")
                    f.write(
                        f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                    f.write(f"\t</topic>\n")

                elif corpus == "2021":
                    f.write(f"\t<topic>\n")
                    f.write(f"\t\t<number>{topic_id}</number>\n")
                    f.write(
                        f"\t\t<query>{topics[topic_id]['title']}</query>\n")
                    f.write(
                        f"\t\t<description>{variants[topic_id][i-1]}</description>\n")
                    f.write(
                        f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                    f.write(
                        f"\t\t<disclaimer>{topics[topic_id]['disclaimer']}</disclaimer>\n")
                    f.write(
                        f"\t\t<stance>{topics[topic_id]['answer']}</stance>\n")
                    f.write(
                        f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                    f.write(f"\t</topic>\n")

                else:     # clef
                    f.write(f"<query>\n")
                    f.write(f"\t\t<id>{topic_id}</id>\n")
                    f.write(f"\t\t<title>{variants[topic_id][i-1]}</title>\n")
                    f.write(
                        f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                    f.write(
                        f"\t\t<originaltitle>{topics[topic_id]['description']}</originaltitle>\n")
                    f.write(f"</query>\n")

            f.write("</queries>\n" if corpus == "clef" else "</topics>\n")


def save_jsonl(variants, filename, n):
    for i in range(1, n + 1):
        with open(f'{filename}_{i}.jsonl', 'w') as f:
            for topic_id in variants:
                json_line = {"_id": topic_id, "text": variants[topic_id][i-1]}
                f.write(json.dumps(json_line) + "\n")


def generate_query_variants(topics, role=True, narrative=True, chain_of_thought=2, n=5):
    variants = {}
    for topic_id in topics:
        print(f"TOPIC_ID: {topic_id}")
        retry = 0
        while retry < 20:
            # original query variantions:
            if query_type == "description":
                prompt = get_prompt_variants(topics[topic_id]['description'], role=role,
                                             narrative=topics[topic_id]['narrative'] if narrative else None,
                                             chain_of_thought=chain_of_thought, n=n)
            else:
                prompt = get_prompt_variants(topics[topic_id]['title'], role=role,
                                             narrative=topics[topic_id]['narrative'] if narrative else None,
                                             chain_of_thought=chain_of_thought, n=n)
            if retry >= 1:
                prompt += "\nUse a list format, as in the example: [\"query variant 1\", \"query variant 2\", ...]"
            print(prompt)
            response = chat_with_gpt4(client, prompt)
            print(response["response"] + "\n")

            # Parse the response to JSON checking for errors
            try:
                # Store the variants in the dictionary
                variants[topic_id] = json.loads(response["response"])
                retry = 50   # Exit the loop

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                if retry == 0:
                    print("Retrying...\n")
                retry += 1

        if retry != 50:
            print("Fatal error")
            exit()

    # Create path if it does not exist
    beginning = "" if topics_type == "original" else "gen_narr_"
    classification = f'{beginning}{topics_type}_{"role" if role else "norole"}_{"narrative" if narrative else "nonarrative"}_chainofth{chain_of_thought}'
    if query_type == "description":
        path = f'query_variants_T07/{corpus}/{classification}'
    else:
        path = f'query_variants_T07/{corpus}/title/{classification}'
    if not os.path.exists(path):
        os.makedirs(path)

    filename = f'{path}/{classification}'
    save_xml(topics, variants, filename, n)
    save_jsonl(variants, filename, n)


def save_scores(scores, filename):
    with open(f'{filename}.json', 'w') as f:
        json.dump(scores, f)


def evaluate_queries(topics, role=True, narrative=True, chain_of_thought=2):
    scores = {}
    for topic_id in topics:
        prompt = get_prompt_evaluation(topics[topic_id]['description'], role=role,
                                       narrative=topics[topic_id]['narrative'] if narrative else None,
                                       chain_of_thought=chain_of_thought)
        print(prompt)
        response = chat_with_gpt4(client, prompt)
        print(response["response"] + "\n")

        # Parse the response to JSON checking for errors
        try:
            json_response = json.loads(response["response"])
            scores[topic_id] = json_response

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

    filename = f'query_scores/query_scores_{"role" if role else "norole"}_{"narrative" if narrative else "nonarrative"}_chainofth{chain_of_thought}'
    save_scores(scores, filename)


def print_prompts():
    user_input = input("Enter the prompt type: [evaluate/variants/narrative] ")
    if user_input.lower() not in ["evaluate", "variants", "narrative"]:
        print("Invalid input. Please try again.")
        return

    if user_input.lower() in ["evaluate", "variants"]:
        user_role = input("Enter the role: [True/False] ")
        user_narrative = input("Enter the narrative: [True/False] ")
        user_chain_of_thought = input("Enter the chain of thought: [0/1/2] ")

        if user_role.lower() not in ["true", "false"] or user_narrative.lower() not in ["true", "false"] or user_chain_of_thought not in ["0", "1", "2"]:
            print("Invalid input. Please try again.")
            return

        parsed_role = True if user_role.lower() == "true" else False
        parsed_narrative = "query_narrative" if user_narrative.lower() == "true" else None
        parsed_chain_of_thought = int(user_chain_of_thought)

    if user_input.lower() in ["evaluate"]:
        prompt = get_prompt_evaluation("query_description", role=parsed_role,
                                       narrative=parsed_narrative, chain_of_thought=parsed_chain_of_thought)
    elif user_input.lower() in ["variants"]:
        prompt = get_prompt_variants("query_description", role=parsed_role,
                                     narrative=parsed_narrative, chain_of_thought=parsed_chain_of_thought, n=5)
    elif user_input.lower() in ["narrative"]:
        user_narrative = input(
            "Which type of narrative? [examples/style/basic/trec] ")
        if user_narrative.lower() in ["examples"]:
            prompt = write_narrative_from_examples("query_description")
        elif user_narrative.lower() in ["style"]:
            prompt = write_narrative_from_style_description(
                "query_description")
        elif user_narrative.lower() in ["basic"]:
            prompt = write_narrative_basic_prompt("query_description")
        elif user_narrative.lower() in ["trec"]:
            prompt = write_narrative_from_TREC("query_description")
        else:
            print("Invalid input. Please try again.")
            return

    print(prompt)


def write_all_narratives(topics, narrative_type):
    if narrative_type == "examples":
        func = write_narrative_from_examples
    elif narrative_type == "style":
        func = write_narrative_from_style_description
    elif narrative_type == "trec":
        func = write_narrative_from_TREC
    else:  # basic
        func = write_narrative_basic_prompt

    if query_type == "description":
        xml_filename = f"topics_with_generated_narratives_from_{narrative_type}_{corpus}.xml"
    else:
        xml_filename = f"topics_with_generated_narratives_from_{narrative_type}_{corpus}_title.xml"

    responses = {}
    for topic_id in topics:
        if query_type == "description":
            prompt = func(topics[topic_id]['description'])
        else:
            prompt = func(topics[topic_id]['title'])
        print(prompt)
        response = chat_with_gpt4(client, prompt)
        responses[topic_id] = response["response"]

        # If the response is not complete (i.e., it does not end with a period), retry
        while not response["response"].endswith("."):
            print("Retrying...")
            response = chat_with_gpt4(client, prompt)
            responses[topic_id] = response["response"]

        print(response["response"] + "\n")
        topics[topic_id]['narrative'] = response["response"]

    with open(xml_filename, 'w') as f:
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        f.write("<topics>\n")
        for topic_id in topics:
            if corpus == "clef":
                f.write(f"<query>\n")
                f.write(f"\t\t<id>{topic_id}</id>\n")
                f.write(
                    f"\t\t<title>{topics[topic_id]['description']}</title>\n")
                f.write(
                    f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                f.write(f"</query>\n")

            elif corpus == "2022":
                f.write(f"\t<topic>\n")
                f.write(f"\t\t<number>{topic_id}</number>\n")
                f.write(
                    f"\t\t<question>{topics[topic_id]['description']}</question>\n")
                f.write(f"\t\t<query>{topics[topic_id]['title']}</query>\n")
                f.write(
                    f"\t\t<background>{topics[topic_id]['narrative']}</background>\n")
                f.write(
                    f"\t\t<disclaimer>{topics[topic_id]['disclaimer']}</disclaimer>\n")
                f.write(f"\t\t<answer>{topics[topic_id]['answer']}</answer>\n")
                f.write(
                    f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                f.write(f"\t</topic>\n")

            elif corpus == "2021":
                f.write(f"\t<topic>\n")
                f.write(f"\t\t<number>{topic_id}</number>\n")
                f.write(f"\t\t<query>{topics[topic_id]['title']}</query>\n")
                f.write(
                    f"\t\t<description>{topics[topic_id]['description']}</description>\n")
                f.write(
                    f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                f.write(
                    f"\t\t<disclaimer>{topics[topic_id]['disclaimer']}</disclaimer>\n")
                f.write(f"\t\t<stance>{topics[topic_id]['answer']}</stance>\n")
                f.write(
                    f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                f.write(f"\t</topic>\n")

            else:     # 2020
                f.write(f"\t<topic>\n")
                f.write(f"\t\t<number>{topic_id}</number>\n")
                f.write(f"\t\t<title>{topics[topic_id]['title']}</title>\n")
                f.write(
                    f"\t\t<description>{topics[topic_id]['description']}</description>\n")
                f.write(f"\t\t<answer>{topics[topic_id]['answer']}</answer>\n")
                f.write(
                    f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                f.write(
                    f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                f.write(f"\t</topic>\n")
        f.write("</topics>\n")


def print_menu():
    print("\nAvailable commands:")
    print("1. evaluate - Evaluate queries")
    print("2. variants - Generate query variants")
    print("3. narrative - Write one narrative for a user-given query (examples, style, basic or trec)")
    print("4. all narratives - Write all narratives (examples, style, basic or trec)")
    print("5. print - Print prompts")
    print("6. chat - Chat with GPT-4")
    print("7. quit - Exit the program")
    print("8. controversy - Determine the level of controversy for all queries")
    print("9. passage writing - Write passages for all queries")


def get_narrative_type():
    while True:
        narrative_type = input(
            "Choose narrative type (examples/style/basic/trec): ").lower()
        if narrative_type in ["examples", "style", "basic", "trec"]:
            return narrative_type
        print("Invalid choice. Please enter 'examples', 'style', 'basic' or 'trec'.")


def get_topics_filename():
    while True:
        corpus = input("Choose corpus (2020/2021/2022/CLEF): ").lower()
        if corpus not in ["2020", "2021", "2022", "clef"]:
            print("Invalid choice. Please enter '2020', '2021', '2022' or 'CLEF'.")
            continue

        query_type = input("Choose query type (description/title): ").lower()
        if query_type not in ["description", "title"]:
            print("Invalid choice. Please enter 'description' or 'title'.")
            continue

        topics_type = input(
            "Choose topics type (original/examples/style/basic/trec): ").lower()
        if topics_type == "original":
            if corpus == "clef":
                return corpus, topics_type, query_type, f'../CLEF/queries2016_corregidas.xml'
            return corpus, topics_type, query_type, f'../TREC_{corpus}_BEIR/original-misinfo-resources-{corpus}/topics/misinfo-{corpus}-topics.xml'
        elif topics_type == "examples":
            return corpus, topics_type, query_type, f'./topics_with_generated_narratives_from_examples_{corpus}.xml'
        elif topics_type == "style":
            return corpus, topics_type, query_type, f'./topics_with_generated_narratives_from_style_{corpus}.xml'
        elif topics_type == "basic":
            return corpus, topics_type, query_type, f'./topics_with_generated_narratives_from_basic_{corpus}.xml'
        elif topics_type == "trec":
            if query_type == "title":
                return corpus, topics_type, query_type, f'./topics_with_generated_narratives_from_trec_{corpus}_title.xml'
            return corpus, topics_type, query_type, f'./topics_with_generated_narratives_from_trec_{corpus}.xml'
        else:
            print(
                "Invalid choice. Please enter 'original', 'examples', 'style', 'basic' or 'trec'.")


def controversy_analysis():
    scores = {}
    njudges = 1
    for topic_id in topics:
        retry = 0

        while retry < 20:
            print(f"TOPIC_ID: {topic_id}")
            field = 'description' if corpus == "clef" else "title"
            prompt = get_prompt_controversy(topics[topic_id][field], judges=njudges)
            print(prompt)
            response = chat_with_gpt4(client, prompt)
            # Parse the response to an integer
            try:
                json_response = json.loads(response["response"])
                print(f"Controversy score: {json_response}\n")
                retry = 50   # Exit the loop
                scores[topic_id] = json_response
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                if retry == 0:
                    print("Retrying...\n")
                retry += 1

    filename = f'controversy_results/controversy_scores_{njudges}judges_{corpus}'
    # Save results into a csv file
    cols_dict = {'index': 'topic'}
    for i in range(njudges):
        cols_dict[i] = f"score{i + 1}"
    df = pd.DataFrame.from_dict(scores, orient='index').reset_index().rename(columns=cols_dict)
    df.to_csv(f'{filename}.csv', index=False)


def controversy_analysis_temp(factors = True):
    scores = {}
    temps = np.linspace(0.2, 0.9, 5)      # array([0.2  , 0.375, 0.55 , 0.725, 0.9  ])
    for topic_id in topics:
        scores[topic_id] = []
        for temp in temps:
            retry = 0

            while retry < 20:
                print(f"TOPIC_ID: {topic_id}")
                field = 'description' if corpus == "clef" else "title"
                prompt = get_prompt_controversy(topics[topic_id][field], factors=factors)
                print(prompt)
                response = chat_with_gpt4(client, prompt, temp=temp)
                # Parse the response to an integer
                try:
                    if factors:
                        resp = json.loads(response["response"])
                    else:
                        resp = int(response["response"])
                    print(f"Controversy score: {resp}\n")
                    retry = 50   # Exit the loop
                    scores[topic_id].append(resp)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    if retry == 0:
                        print("Retrying...\n")
                    retry += 1

    filename = f'controversy_results/controversy_scores_factors_{corpus}'
    # Save results into a csv file
    cols_dict = {'index': 'topic'}
    for i in range(len(temps)):
        cols_dict[i] = f"score{i + 1}"
    df = pd.DataFrame.from_dict(scores, orient='index').reset_index().rename(columns=cols_dict)
    df.to_csv(f'{filename}.csv', index=False)

def passage_writing(n=10):
    all_passages = {}
    for topic_id in topics:
        retry = 0

        while retry < 20:
            print(f"TOPIC_ID: {topic_id}")
            prompt = get_passage_writing_prompt(topics[topic_id]["description" if corpus=="clef" else "title"])
            print(prompt)
            response = chat_with_gpt4(client, prompt)

            try:
                passages = response["response"].split("||PAS||")
                passages = [str(p).strip() for p in passages if str(p).strip() != ""]
                print(passages)
                all_passages[topic_id] = passages
                retry = 50
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                if retry == 0:
                    print("Retrying...\n")
                retry += 1

    filename = f'generated_passages/passages'
    # Save results into a csv file
    df = pd.DataFrame.from_dict(all_passages, orient='index').reset_index().rename(columns={'index': 'topic'})
    df.to_csv(f'{filename}.csv', index=False)   

def main():
    parser = configparser.ConfigParser()
    parser.read("config.ini")

    while True:
        print_menu()

        user_input = input("Give instructions: ")
        if user_input.lower() in ["1", "evaluate"]:
            evaluate_queries(topics, role=True,
                             narrative=True, chain_of_thought=0)

        elif user_input.lower() in ["2", "variants"]:
            generate_query_variants(
                topics, role=True, narrative=True, chain_of_thought=1, n=10)

        elif user_input in ["3", "narrative"]:
            narrative_type = get_narrative_type()
            query_description = input("Enter the query description: ")

            if narrative_type == "examples":
                prompt = write_narrative_from_examples(query_description)
            elif narrative_type == "style":
                prompt = write_narrative_from_style_description(
                    query_description)
            elif narrative_type == "trec":
                prompt = write_narrative_from_TREC(query_description)
            else:  # basic
                prompt = write_narrative_basic_prompt(query_description)

            response = chat_with_gpt4(client, prompt)
            print(response["response"])

        elif user_input in ["all narratives", "4"]:
            write_all_narratives(topics, get_narrative_type())

        elif user_input.lower() in ["5", "print"]:
            print_prompts()

        elif user_input.lower() in ["6", "chat"]:
            user_prompt = input("Enter your prompt: ")
            response = chat_with_gpt4(client, user_prompt)
            print(response["response"])

        elif user_input.lower() in ["7", "quit"]:
            print("Assistant: Goodbye!")
            break
        
        elif user_input.lower() in ["8", "controversy"]:
            #controversy_analysis()
            controversy_analysis_temp()

        elif user_input.lower() in ["9", "passage"]:
            passage_writing()

        else:
            print("Invalid command. Please try again.")


# Main program loop
if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read("config.ini")

    api_key = parser.get("OPENAI", "API_KEY")
    client = OpenAI(api_key=api_key)

    corpus, topics_type, query_type, topics_filename = get_topics_filename()
    topics = fetch_topics(topics_filename, corpus)

    main()
