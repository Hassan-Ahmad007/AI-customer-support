import os

import streamlit as st
from crewai import Agent, LLM

from tools import (
    company_knowledge_search,
    order_lookup,
)

from escalation import escalate_to_human


def get_llm():
    api_key = st.secrets["GEMINI_API_KEY"]

    os.environ["GEMINI_API_KEY"] = api_key

    return LLM(
        model="gemini/gemini-3.5-flash-lite",
        api_key=api_key,
        temperature=0.2,
    )


def create_support_agent():

    llm = get_llm()

    return Agent(
        role="Customer Support Specialist",

        goal=(
            "Help customers solve their problems using the "
            "company knowledge base and order information. "
            "Provide accurate, clear, and helpful responses. "
            "Escalate unresolved issues to a human representative."
        ),

        backstory=(
            "You are a professional customer support specialist. "
            "You help customers with company policies, products, "
            "orders, shipping, returns, refunds, troubleshooting, "
            "and other support questions. "
            "You never invent company policies or order information. "
            "When information is unavailable or the issue requires "
            "human intervention, you escalate the request."
        ),

        tools=[
            company_knowledge_search,
            order_lookup,
            escalate_to_human,
        ],

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )
