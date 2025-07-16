# Ragas-Studio
This repository contains my Studio_RAGAS notebook, which focuses on:

RAGAS Framework or Analysis: An in-depth analysis or implementation of the RAGAS (Retrieval-Augmented Generation Assessment) method, if applicable.

Studio Integration: Shows how the method integrates with studio or local environments.

Data Pipeline: Any data preprocessing steps tailored for text, embeddings, or vector stores.

Model Use: Demonstrates how to connect and use an LLM or retrieval pipeline (if relevant).

Results Interpretation: Evaluation metrics or visualization of the system’s performance.

## In this final version of the notebook, I experimented 'mistralai/Mistral-7B-Instruct-v0.1' model with modified prompts, modified human generated questions set, different model validation techniques.
# Set-up the environment
import os
os.kill(os.getpid(), 9)
%%capture
!pip install openai==1.55.3 httpx==0.27.2 --force-reinstall --quiet
# Clone necessary data repository containing document and vector information
!git clone https://github.com/ricklon/mpi_data.git
# Install required Python packages for document processing and language modeling
!pip install -q langchain pypdf langchain_community sentence_transformers faiss-cpu pandas tqdm
# Import necessary libraries
import os
import  pandas as pd
from io import StringIO
from tqdm import tqdm
import time
from  datetime import datetime

# Import LangChain components for constructing the AI pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint

from langchain.document_loaders import PyPDFDirectoryLoader

# Import Google Colab utilities for accessing user-specific data
from google.colab import userdata

# Set up the environment for Hugging Face model access
HUGGINGFACEHUB_API_TOKEN = userdata.get('HG_KEY')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
# Initialize directories and file paths for processing and outputs
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

OUTPUT_DIR = "./out"

# Check if the directory exists, and if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Directory '{OUTPUT_DIR}' was created.")
else:
    print(f"Directory '{OUTPUT_DIR}' already exists.")
# Define the vector source file
# Define file paths for vectors, documents, and questions
VECTOR_SOURCE_FILE="/content/mpi_data/data/out/docs_vectors_2024-03-25_18-48-07"
PAGES_SOURCE_FILE="/content/mpi_data/data/out/docs_split_2024-03-25_18-47-59.jsonl"
QUESTIONS_SOURCE_FILE="/content/mpi_data/data/out/questions_df_2024-03-25_18-55-49"
log_file_path = OUTPUT_DIR
PAPER_SOURCES = "/content/mpi_data/data/paper_sources.csv"
# Set-up the Environment for embedding model
# Initialize models and vector storage for document processing
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
HG_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# hg_model = "HuggingFaceH4/zephyr-7b-beta"
#hg_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" ## Tiny LLAMA not supported

#Conservative parameters: Ensures precise, deterministic, and highly controlled output with minimal randomness.
llm = HuggingFaceEndpoint(
    repo_id=HG_MODEL,
    max_new_tokens=200,  # the maximum length of the new text to be generated
    top_k=5,  # the number of highest probability vocabulary tokens to keep for top-k-filtering
    top_p=0.90,  # if set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation
    typical_p=0.90,  # used in nucleus sampling, must be in [0.0, 1.0]
    temperature=0.02,  # the value used to module the next token probabilities
    repetition_penalty=1.2  # the parameter for repetition penalty
    )



embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
)
# Run the LLM
# Load the FAISS index
vectorstore = FAISS.load_local(VECTOR_SOURCE_FILE, embeddings=embedding_model, allow_dangerous_deserialization=True)

# Configure the retrieval and query processing chain
retriever = vectorstore.as_retriever()
#Prompt template
template = """
You are a researcher and professor in mining and geology. Given the comprehensive documents provided, including technical reports and feasibility studies:
{context}

Question: {question}
Please provide precise and specific answers based on the context.
"""
prompt = ChatPromptTemplate.from_template(template)

#query processing chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
#Execute the chain for a specific question and log the results
try:
    result = chain.invoke("What is the estimated reserve of the Red Dog mine as described in the document?", timeout=300)
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")

#chain.invoke("What is the estimated reserve of the Red Dog mine as described in the document?", timeout=300)
# Test the data collection
# Create a CSV from a string of question data and process each question using the configured chain
# Diagnostic Questions

csv_data = """
"Question","Purpose"
"What is the estimated reserve of the Red Dog mine as described in the document?","To verify the document's data on mineral reserves."
"What are the projected operational costs for the Arctic mine according to the feasibility study?","To validate financial projections provided in the document."
"What mining method is proposed for the Hermosa mine in its report?","To confirm technical planning details."
"What are the environmental considerations for the Palmer mine as detailed in the technical report?","To check for environmental impact assessments and mitigation strategies."
"What is the expected lifespan of the Strong and Harris mine based on its preliminary economic assessment?","To validate the mine's economic viability."
"How does the Greens Creek mine manage waste materials, according to the technical report?","To ensure responsible waste management practices are described."
"What exploration techniques were used to determine the extent of the Empire Mine's deposits?","To verify exploration methods and their effectiveness."
"""

# Use StringIO to simulate a file object
data = StringIO(csv_data)

# Read the CSV data into a pandas DataFrame
questions_df = pd.read_csv(data, quotechar='"')

questions_df.to_csv(f"{log_file_path}/questions_df_{start_time}", index=False)


#"How does the Revenue-Virginius mine's geology support its classification as an Intermediate-sulfidation epithermal deposit?","To confirm geological characteristics specified in the report."
#"How does the Pickett Mountain mine's deposit type influence its mining technique?","To understand the correlation between deposit type and mining methodology."
#"What makes the Elk Creek mine's Carbonatite niobium deposit unique, as detailed in the annual report?","To confirm unique geological features."

# Read the document source spreadsheet pandas df
# Create an empty DataFrame
results_df = pd.DataFrame(columns=['Question', 'Answer', 'Timestamp', 'ExecutionTime'])

#Loop over Questions:
for q in tqdm(questions_df['Question'], desc="Processing Questions"):
    start_time = time.time()  # Start timing
    answer = chain.invoke(q)  # Invoke the chain to get the answer
    execution_time = time.time() - start_time  # Calculate execution time
    timestamp = datetime.now()  # Get the current timestamp

   # Prepare the new data as a DataFrame
    new_data = pd.DataFrame({
        'Question': [q],
        'Answer': [answer],
        'Timestamp': [timestamp],
        'ExecutionTime': [execution_time]
    })

    # Concatenate the new data with the existing DataFrame
    results_df = pd.concat([results_df, new_data], ignore_index=True)
#Exporting DataFrame to CSV
results_df.to_csv(f"{log_file_path}/results_df_{start_time}", index=False)
# Show the Results
#Display the First Few Rows of DataFrame
results_df.head()
#Accessing a Specific question and answer
results_df.iloc[1]["Answer"]

## Time Sensitive Questions Set
# Define file paths for time sensitive questions
#QUESTIONS_SOURCE_FILE="/content/time_sensative_questions.xlsx"
#Execute the chain for a specific question and log the results
chain.invoke("When did the modern-era exploration of the Strong and Harris project commence?")
# Test the data collection
# Create a CSV from a string of question data and process each question using the configured chain
# Diagnostic Questions

csv_data = """
"Question","Purpose"
"When was the first significant discovery made in Pickett Mountain property area?"
"When did the modern-era exploration of the Strong and Harris project commence?"
"What is the production from the Red Dog Mine through 2016?"
"When the three test programs were completed by SGS Mineral Services (SGS)?"
"What is the current amount filed for Financial Assurance Obligation with the State of Alaska for Red Dog Mine?"
"What is the estimated annual requirement for water treatment at Red Dog Mine post-closure?"
"As of what date are the mineral resources for the Taylor Deposit reported?"
"""

# Use StringIO to simulate a file object
data = StringIO(csv_data)

# Read the CSV data into a pandas DataFrame
questions_df = pd.read_csv(data, quotechar='"')

questions_df.to_csv(f"{log_file_path}/questions_df_{start_time}", index=False)
# Read the document source spreadsheet pandas df
# Create an empty DataFrame
results_df = pd.DataFrame(columns=['Question', 'Answer', 'Timestamp', 'ExecutionTime'])

#Loop over Questions:
for q in tqdm(questions_df['Question'], desc="Processing Questions"):
    start_time = time.time()  # Start timing
    answer = chain.invoke(q)  # Invoke the chain to get the answer
    execution_time = time.time() - start_time  # Calculate execution time
    timestamp = datetime.now()  # Get the current timestamp

   # Prepare the new data as a DataFrame
    new_data = pd.DataFrame({
        'Question': [q],
        'Answer': [answer],
        'Timestamp': [timestamp],
        'ExecutionTime': [execution_time]
    })

    # Concatenate the new data with the existing DataFrame
    results_df = pd.concat([results_df, new_data], ignore_index=True)
#Exporting DataFrame to CSV
results_df.to_csv(f"{log_file_path}/results_df_{start_time}", index=False)
#Display the First Few Rows of DataFrame
results_df.head()
#Accessing a Specific question and answer
results_df.iloc[1]["Answer"]
## Evaluation of the RAG Model
# Combine all the results in one data frame
#Change the file name everytime the file is created
#copy and paste the filepath from content/output
results1 = pd.read_csv('/content/out/results_df_1733619997.2718027')  #copy and paste the filepath from content/output
#results2 = pd.read_csv('/content/out/results_df_1733537131.9020996')
#Checking the results 1 or 2
results1.head()
# Combine data frames into one
#Combined_df = pd.concat([results1, results2], ignore_index=True)
Combined_df = results1
Combined_df
Combined_df.shape
# Load the ground truth data
ground_df = pd.read_excel('/ground_truth_2.xlsx')
ground_df.head()
# Renaming the column in ground_df for consistency
ground_df.rename(columns={'question': 'Question'}, inplace=True)
# Renaming the column in ground_df for differentiation
ground_df.rename(columns={'answer': 'Original Answer'}, inplace=True)
# Merging the DataFrames
Combined_df = pd.merge(Combined_df, ground_df[['Question', 'Original Answer', 'contexts']], on='Question', how='left')
Combined_df
# Create a new 'Prompt' column with empty strings
Combined_df['Different_Questions_Set'] = ''

# Assign values for the first 10 rows as 'Time Sensative Questions'
Combined_df.loc[:6, 'Different_Questions_Set'] = 'Regular_Questions'

# Assign values for the 7 rows as 'Regular Questions'
Combined_df.loc[7:13, 'Different_Questions_Set'] = 'Time_Sensative_Questions'
# Check the first few rows to confirm the merge
Combined_df
## Evaluation Method 1: SpaCy for Advanced Textual Comparisons
#Installing spaCy
!pip install spacy

#Downloading a spaCy Language Model
!python -m spacy download en_core_web_md

# Step 1: Import spaCy
import spacy

# Step 2: Load the pre-trained model
nlp = spacy.load('en_core_web_md')

# Step 3: Test the model
doc = nlp("Hello, world!")
for token in doc:
    print(token.text, token.lemma_, token.pos_)

# Load a pre-trained model
nlp = spacy.load('en_core_web_md')  # Make sure to download this model first

# Compute similarities
def compute_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

#Compute Similarity for Each Row
Combined_df['similarity'] = Combined_df.apply(lambda row: compute_similarity(row['Original Answer'], row['Answer']), axis=1)

#Print Average Similarity
print("Average similarity:", Combined_df['similarity'].mean())
#Describe the 'similarity' score for all questions
Combined_df['similarity']
## Evaluation Method 2: Hugging Face's Transformers for Contextual Embeddings (Cosine Similarity)
#Importing Libraries and Loading Models
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

#Function to Get Embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# Compute cosine similarity for embeddings
from scipy.spatial.distance import cosine

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

Combined_df['cosine_similarity'] = Combined_df.apply(lambda row: cosine_similarity(get_embedding(row['Original Answer']), get_embedding(row['Answer'])), axis=1)

#Print average similarity
print("Average cosine similarity:", Combined_df['cosine_similarity'].mean())
#Describe the 'similarity' score for all questions
Combined_df['cosine_similarity']
#Evaluation Method 3: BLEU Score (Bilingual Evaluation Understudy)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    # Using smoothing function
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    return score

# Apply BLEU with smoothing to DataFrame
Combined_df['bleu_score'] = Combined_df.apply(
    lambda row: calculate_bleu(row['Original Answer'], row['Answer']), axis=1
)
#Print Average Similarity
print("Average similarity:", Combined_df['bleu_score'].mean())
#Describe the 'similarity' score for all questions
Combined_df['bleu_score']
#Evaluation Method 4: Jaccard Similarity

def jaccard_similarity(query, document):
    intersection = set(query.split()).intersection(set(document.split()))
    union = set(query.split()).union(set(document.split()))
    return len(intersection) / len(union)

# Apply Jaccard to DataFrame
Combined_df['jaccard_similarity'] = Combined_df.apply(lambda row: jaccard_similarity(row['Original Answer'], row['Answer']), axis=1)

#Print Average Similarity
print("Average similarity:", Combined_df['jaccard_similarity'].mean())
#Describe the 'similarity' score for all questions
Combined_df['jaccard_similarity']
#Evaluation Method 5: BERTScore
# Install the bert_score library
!pip install bert-score

from bert_score import score
def calculate_bertscore(Combined_df):
    # Define a max length (512 tokens for roberta-large model)
    max_length = 512

    # Helper function to truncate text to avoid tokenization issues
    def truncate_text(text, tokenizer, max_length=max_length):
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length:
            return tokenizer.convert_tokens_to_string(tokens[:max_length])
        return text

    # Initialize the tokenizer separately to use for truncation
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    # Preprocess each text by truncating if necessary
    Combined_df['Truncated Original Answer'] = Combined_df['Original Answer'].apply(lambda x: truncate_text(str(x), tokenizer))
    Combined_df['Truncated Answer'] = Combined_df['Answer'].apply(lambda x: truncate_text(str(x), tokenizer))

    # Calculate BERTScore and handle potential errors with try-except
    try:
        P, R, F1 = score(Combined_df['Truncated Answer'].tolist(), Combined_df['Truncated Original Answer'].tolist(), lang="en")
        Combined_df['bertscore_P'] = P.cpu().detach().numpy().tolist()
        Combined_df['bertscore_R'] = R.cpu().detach().numpy().tolist()
        Combined_df['bertscore_F1'] = F1.cpu().detach().numpy().tolist()
    except KeyError as e:
        print(f"KeyError encountered for text: {e}")

    return Combined_df

# Apply the BERTScore function
Combined_df = calculate_bertscore(Combined_df)
#Print Average Similarity
print("Average similarity:", Combined_df['bertscore_F1'].mean())
#Describe the 'similarity' score for all questions
Combined_df['bertscore_F1']
#6. RAGAS
pip install ragas

import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from transformers import pipeline
from langchain import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
import os
from google.colab import userdata
from huggingface_hub import login


HUGGINGFACEHUB_API_TOKEN = userdata.get('HG_KEY')
os.environ["HUGGINGFACEHUB_API_TOKEN"]  = HUGGINGFACEHUB_API_TOKEN

login(token=HUGGINGFACEHUB_API_TOKEN , add_to_git_credential=True)

# Set OpenAI API key
#os.environ["OPENAI_API_KEY"] = "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"
# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
# Configure the LLM
llm_pipeline = pipeline(
    "text2text-generation",
    model="meta-llama/Llama-3.2-1b",  # Replace with your chosen model
    max_new_tokens=512,
    #device=0  # Use GPU if available
    device="cpu"  # Use CPU if GPU is not available
)
local_llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=llm_pipeline))
# Configure the embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
local_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# Clean input data
def clean_text(text):
    return text.strip().replace("\n", " ").replace("\t", " ")
# Prepare evaluation data
eval_data = [
    {
        "user_input": row["Question"],  # Input question
        "response": row["Answer"],  # Model's generated response
        "retrieved_contexts": [row["contexts"]],  # Placeholder for retrieved context; replace with actual if available
        "reference": row["Original Answer"],  # Ground truth or expected answer
    }
    for _, row in Combined_df.iterrows()
]

# Retrieve similar documents based on the user query
def get_text_from_faiss(user_input, vectorstore, k=1):
    try:
        # Perform similarity search for the user input
        results = vectorstore.similarity_search(user_input, k=k)
        return [res.page_content for res in results]
    except Exception as e:
        return [f"Error retrieving document: {e}"]


for entry in eval_data:
    entry["retrieved_contexts"] = get_text_from_faiss(entry["user_input"], vectorstore, k=3)  # Retrieve top 3 matches

#Verify the Updated Data
for entry in eval_data[:5]:  # Inspect the first 5 entries
    print(entry)

# Create the EvaluationDataset
eval_dataset = EvaluationDataset.from_list(eval_data)
# Convert eval_data to a DataFrame
eval_data_df = pd.DataFrame(eval_data)


# Convert to a DataFrame for inspection
print(eval_data_df.head(10))  # Display the first 10 rows
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
import openai

import os
import openai

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-DzABB3tgIHaUEwY5qhWuT3BlbkFJvuGB9b95SelQ19aE23Ha"
openai.api_key = os.getenv("OPENAI_API_KEY")


# Step 1: Initialize the LLM and Embeddings
# Replace "gpt-4o-mini" with the actual model ID for GPT-4o Mini
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# Initialize OpenAI embeddings
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Step 2: Define Metrics
metrics = [
    LLMContextRecall(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]

from ragas import EvaluationDataset

# Convert eval_data (list) to EvaluationDataset
eval_dataset = EvaluationDataset.from_list(eval_data)

# Step 3: Evaluate the Dataset
results = evaluate(dataset=eval_dataset, metrics=metrics)

# Step 4: Convert Results to a DataFrame
df = results.to_pandas()
print(df.head())

# Save the DataFrame as an Excel file
excel_file_path = "evaluation_results.xlsx"
df.to_excel(excel_file_path, index=False)

print(f"Results have been saved to {excel_file_path}")
# Compute the Average Scores for RAGAS Metrics
# For regular questions (first 10 rows)
regular_questions = df.iloc[:10]  # Select the first 10 rows
regular_scores = regular_questions.mean(numeric_only=True)

# For time-sensitive questions (last 7 rows)
time_sensitive_questions = df.iloc[-7:]  # Select the last 7 rows
time_sensitive_scores = time_sensitive_questions.mean(numeric_only=True)

# Compute a single RAGAS score for each category
average_ragas_score_regular = regular_scores.mean()
average_ragas_score_time_sensitive = time_sensitive_scores.mean()

# Print the results
print("Average Scores for Regular Questions:")
print(regular_scores)
print(f"Average RAGAS Score for Regular Questions: {single_ragas_score_regular}")

print("\nAverage Scores for Time-Sensitive Questions:")
print(time_sensitive_scores)
print(f"Average RAGAS Score for Time-Sensitive Questions: {single_ragas_score_time_sensitive}")



#Combine and comparison of the results
# Calculate the average of metrics grouped by 'Different_Questions_Set'
average_scores = Combined_df.groupby('Different_Questions_Set')[['similarity', 'cosine_similarity', 'bleu_score', 'jaccard_similarity', 'bertscore_F1']].mean()

# Add single RAGAS scores for regular and time-sensitive questions
average_ragas_regular = average_ragas_score_regular  # Single RAGAS score for regular questions
average_ragas_time_sensitive = average_ragas_score_time_sensitive  # Single RAGAS score for time-sensitive questions

# Create a new DataFrame to include RAGAS scores
ragas_scores_df = pd.DataFrame({
    "Different_Questions_Set": ["Regular Questions", "Time-Sensitive Questions"],
    "Average_RAGAS_Score": [average_ragas_regular, average_ragas_time_sensitive]
})

# Merge with the average_scores DataFrame for comparison
average_scores = average_scores.reset_index()
comparison_df = pd.concat([average_scores, ragas_scores_df], ignore_index=True)

# Display the resulting DataFrame
print("Comparison of Metrics with average RAGAS Scores:")
print(comparison_df)

# Save to an Excel file for detailed analysis
comparison_df.to_excel("comparison_with_ragas_scores.xlsx", index=False)
print("Comparison results saved to comparison_with_ragas_scores.xlsx")

# Grouping and calculating average scores for metrics by question type
average_scores = (
    Combined_df.groupby("Different_Questions_Set")[
        ["similarity", "cosine_similarity", "bleu_score", "jaccard_similarity", "bertscore_F1"]
    ]
    .mean()
    .fillna(0)  # Replace NaN values with 0
)

# Adding the Average RAGAS Score to the table
average_scores["Average_RAGAS_Score"] = [
    average_ragas_regular, average_ragas_time_sensitive
]

# Saving results to a new DataFrame for better formatting
comparison_results = average_scores.reset_index()

# Displaying the comparison results
print("Comparison of Metrics with Average RAGAS Scores:")
print(comparison_results)

# Saving the results to an Excel file
comparison_results.to_excel("comparison_with_ragas_scores_cleaned.xlsx", index=False)
print("Cleaned comparison results saved to comparison_with_ragas_scores_cleaned.xlsx")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas.embeddings import HuggingFaceEmbeddingsWrapper
from ragas import evaluate

# Step 1: Initialize Hugging Face Model and Tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Create a Hugging Face Pipeline
pipeline_model = pipeline("text-generation", model=model, tokenizer=tokenizer)
evaluator_llm = HuggingFacePipeline(pipeline=pipeline_model)

# Step 3: Initialize Hugging Face Embeddings
huggingface_embeddings = HuggingFaceEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# Step 4: Define Metrics
metrics = [
    LLMContextRecall(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=huggingface_embeddings)
]

# Step 5: Evaluate Dataset
results = evaluate(dataset=eval_dataset, metrics=metrics)

# Step 6: Convert Results to DataFrame and Analyze
df = results.to_pandas()
average_scores = df.mean(numeric_only=True)
print("Average Scores for RAGAS Metrics with Hugging Face Models:")
print(average_scores)

# Visualizations
