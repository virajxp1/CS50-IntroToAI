Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. 

This question answering system will perform two tasks: document retrieval and passage retrieval. 

Our system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined.

To find the most relevant documents, we’ll use tf-idf to rank documents based both on term frequency for words in the query as well as inverse document frequency for words in the query. We use a combination of inverse document frequency and a query term density measure to determine which sentence is most relevent.
