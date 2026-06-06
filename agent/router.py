def route_query(question):

    question = question.lower()

    rag_keywords = [
        "policy",
        "document",
        "handbook",
        "onboarding",
        "leave",
        "proposal",
        "uploaded",
        "hr",
        "company"
    ]

    for word in rag_keywords:
        if word in question:
            return "RAG"

    return "GENERAL"