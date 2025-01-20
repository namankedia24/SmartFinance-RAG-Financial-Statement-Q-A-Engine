import time
from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode
import os
import re
from openai import OpenAI
from llama_index.core.schema import TextNode
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import VECTOR_STORE_DIR, EMBED_MODEL_NAME, LLM_MODEL_NAME
from together import Together


# import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RETRIEVER_SIMILARITY_TOP_K: int = 5
RERANKER_CHOICE_BATCH_SIZE: int = 3

RERANKER_TOP_N: int = 1

# TODO: implement RAG pipline with metadata filtering and reranking.
# When initializing different object in the constructor, please make
# the objects protected, which is achieved by underscoring the beginning
# of the object, as in the example with the embedding model:
# self._embed_model = ...

# We suggest sticking to the provided template for we believe it to be
# the simplest implementation way. Please, provide explanation if you
# find it necessary to change template.


def calculate_time(func):
    
    def inner1(*args, **kwargs):

        # storing time before function execution
        begin = time.time()
        
        val = func(*args, **kwargs)

        # storing time after function execution
        end = time.time()
        print("Total time taken in : ", func.__name__, end - begin)
        return val

    return inner1

class RAGSystem:
    def __init__(self):
        # Initialize embedding model given EMBED_MODEL_NAME.

        self._embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
        # Initialize a vector database from the existing collection
        # in VECTOR_STORE_DIR and a corresponding vector store index.
        # We recommend ChromaDB. Embedding model should be the same
        # as for storing the nodes.
        self.client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        collection_name = "final"
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Initialize LLM given LLM_MODEL_NAME.
        # Tip: for the pipeline to work correctly, it is likely
        # you will need to create a tokenizer for the model.
        # We suggest looking into AutoTokenizer.
        
        togetherai_api_key = 'dd7aaa683f7a978000e2f6b9bb6df2a3ab1d7ecb7dc0aa248c989c1118a0c463'
        self.client_together = OpenAI(api_key=togetherai_api_key,
                base_url='https://api.together.xyz')
        # self.client_together = Together(api_key=togetherai_api_key)

        # Initialize reranker. We suggest LLM-based reranker with
        # RERANKER_CHOICE_BATCH_SIZE nodes to consider from the retriever and
        # RERANKER_TOP_N documents to return.
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # @calculate_time
    def respond(self, query: str, ticker: str, year: str) -> tuple[str, TextNode]:
        """
        Given a questy, a ticker and a year, should return a response
        to the provided query for the given company in the given year and
        the most relevant node.
        """
        # Convert the given query string to a query bundle (most likely
        # required for correct work).
        # query_bundle = QueryBundle(query)

        query_embedding = self._embed_model.encode(query).tolist()

        # Initialize metadata filters.
        # metadata_filter = {"ticker": ticker, "year": year}

        metadata_filter = {
            "$and": [
                {"ticker": {"$eq": ticker}},
                {"year": {"$eq": year}}
                ]
        }           

        # Initialize retriever from the vector store index
        # with the filters above and RETRIEVER_SIMILARITY_TOP_K.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=RETRIEVER_SIMILARITY_TOP_K,
            where=metadata_filter,
            include=["metadatas", "documents", "distances"]
        )
        
        # Retrieve the nodes for the provided query.
        retrieved_nodes = [
            TextNode(text=document, metadata=results["metadatas"][0][i])
            for i, document in enumerate(results["documents"][0])
        ]

        # print("Retrieved nodes:" + str(retrieved_nodes))
        # print(results['metadatas'][0])
        # Apply the reranker to the retrieved nodes and get
        # the best one.
        reranked_nodes = self.rerank(query, retrieved_nodes[:RERANKER_CHOICE_BATCH_SIZE])
        selected_node = reranked_nodes[0] if reranked_nodes else None
        print("Best retrieved node:" + str(selected_node.text) if selected_node else "No relevant context found.")

        # Merge the reranked nodes with the query and give it
        # to the LLM to get the response.
        # TODO
        llm_input_text = f"Answer the question given the context when possible. Keep in mind to return correct units when returning numerical data. Question: {query}\nContext: {selected_node.text if selected_node else 'No relevant context found.'}\nAnswer:"
    
        prompt_json = [{'role': 'user', 'content': llm_input_text}]
        chat_completion = self.client_together.chat.completions.create(model=LLM_MODEL_NAME,
                                                          messages=prompt_json,
                                                          temperature=0,
                                                          )
        response = chat_completion.choices[0].message.content
        return response, selected_node
    
    def rerank(self, query: str, retrieved_nodes: list) -> list:
        if not retrieved_nodes:
            return []
            
        pairs = [[query, node.text] for node in retrieved_nodes]
        
        scores = self.cross_encoder.predict(pairs)
        
        scored_nodes = list(zip(retrieved_nodes, scores))
        ranked_nodes = [node for node, score in sorted(scored_nodes, 
                                                     key=lambda x: -x[1])]
        
        return ranked_nodes[:RERANKER_TOP_N]


if __name__ == '__main__':
    # TODO: come up with 2 questions for each ticker based on the documents,
    # and evaluate the quality of both the retriever by the most relevant node provided
    # and the response. Implementing automatic evaluation techniques is not required,
    # we expect to see detailed analysis of the manual evaluation.
    rag_system = RAGSystem()

    # Example questions for evaluation
    questions = [
    ("What was Boeing's revenue in 2014?", "BA", "2015"),
    ("What were Boeing's net earnings in 2012?", "BA", "2013"),
    ("What was Boeing's commercial airplanes segment revenue in 2013?", "BA", "2014"),
    ("How much revenue did Boeing's Defense segment generate in 2014?", "BA", "2015"),
    ("What were Boeing's operating expenses in 2011?", "BA", "2012"),

    ("What was Biogen's total revenue in 2015?", "BIIB", "2016"),
    ("How much did Biogen spend on R&D in 2014?", "BIIB", "2015"),
    ("What were Biogen's product sales in 2013?", "BIIB", "2014"),
    ("What was Biogen's operating income in 2016?", "BIIB", "2017"),
    ("How much cash did Biogen have at the end of 2012?", "BIIB", "2013"),

    ("What was Dominion's operating revenue in 2014?", "D", "2015"),
    ("How much were Dominion's electric utility sales in 2013?", "D", "2014"),
    ("What were Dominion's capital expenditures in 2015?", "D", "2016"),
    ("What was Dominion's net income in 2012?", "D", "2013"),
    ("How much debt did Dominion have in 2016?", "D", "2017"),

    ("What was F5's product revenue in 2014?", "FFIV", "2015"),
    ("How much service revenue did F5 generate in 2013?", "FFIV", "2014"),
    ("What were F5's total operating expenses in 2015?", "FFIV", "2016"),
    ("What was F5's gross profit in 2012?", "FFIV", "2013"),
    ("How much did F5 spend on sales and marketing in 2016?", "FFIV", "2017"),

    ("What was Google's advertising revenue in 2015?", "GOOG", "2016"),
    ("How much revenue did Google generate from other sources in 2013?", "GOOG", "2014"),
    ("What were Google's R&D expenses in 2015?", "GOOG", "2016"),
    ("What was Google's total revenue in 2017?", "GOOG", "2017"),
    ("How much did Google spend on traffic acquisition costs in 2016?", "GOOG", "2017"),

    ("What was Hilton's total revenue in 2014?", "HLT", "2015"),
    ("How much revenue came from owned hotels in 2013?", "HLT", "2014"),
    ("What was Hilton's franchise fee revenue in 2015?", "HLT", "2016"),
    ("What were Hilton's operating expenses in 2012?", "HLT", "2013"),
    ("How much management fee revenue did Hilton generate in 2016?", "HLT", "2017"),

    ("What was Hormel's net sales in 2014?", "HRL", "2015"),
    ("How much was Hormel's grocery products revenue in 2013?", "HRL", "2014"),
    ("What were Hormel's refrigerated foods sales in 2015?", "HRL", "2016"),
    ("What was Hormel's net earnings in 2012?", "HRL", "2013"),
    ("How much did Hormel spend on advertising in 2016?", "HRL", "2017"),

    ("What was IDEXX's total revenue in 2014?", "IDXX", "2015"),
    ("How much came from companion animal diagnostics in 2013?", "IDXX", "2014"),
    ("What were IDEXX's laboratory revenue in 2015?", "IDXX", "2016"),
    ("What was IDEXX's operating income in 2012?", "IDXX", "2013"),
    ("How much did IDEXX spend on R&D in 2016?", "IDXX", "2017"),

    ("What was Constellation's wine sales in 2014?", "STZ", "2015"),
    ("How much beer revenue did Constellation generate in 2013?", "STZ", "2014"),
    ("What were Constellation's spirits sales in 2015?", "STZ", "2016"),
    ("What was Constellation's operating income in 2012?", "STZ", "2013"),
    ("How much did Constellation spend on marketing in 2016?", "STZ", "2017"),

    ("What was Tesla's automotive revenue in 2014?", "TSLA", "2015"),
    ("How much revenue came from regulatory credits in 2013?", "TSLA", "2014"),
    ("What were Tesla's R&D expenses in 2015?", "TSLA", "2016"),
    ("What was Tesla's gross profit in 2012?", "TSLA", "2013"),
    ("How much did Tesla spend on capital expenditures in 2016?", "TSLA", "2017")
    ]

    for question, ticker, year in questions:
        print(f"Query: {question}")
        
        response, relevant_node = rag_system.respond(question, ticker, year)
        
        print(f"Response: {response}")
        print("\n\n")
        
        # if relevant_node:
        #     print(f"Relevant Node: {relevant_node.text}")

    pass


# Possible scores:
# [20 pts]        Basic RAG implemented, without metadata filtering
#                 and reranking.
# [+10 pts]       Metadata filtering implemented.
# [+5 pts]        Reranking implemented.
# [up to +10 pts] Manual evaluation implemented and analyzed.
