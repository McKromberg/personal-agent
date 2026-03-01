#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 23:39:40 2025
@author: maximilianromberg
"""

import os
from langfuse import get_client
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler


class PlanningAgent:

    def __init__(self, model: str, callback_handler: CallbackHandler):
        self.model = model
        self.llm = None
        self.agent = None
        self.langfuse_handler = callback_handler

    async def initialize(self):
        """Async initialization - call this after creating the instance."""
        print("Initializing Planning Agent")
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY")
        )
        self.agent = await self._create_agent()

    async def _create_agent(self):
    
        #tools = await load_mcp_tools(session)
        # Agent prompt
        prompt = """
You are a planning agent in a multi agent system.
Do as you are told.
"""
        # Build agent
        agent = create_agent(
            model=self.llm,
            #tools=tools,
            system_prompt=prompt
        )
        return agent

    async def invoke_agent(self, instruction: str) -> str:
        """
        Invoke the planning agent and return the last message.
        
        Args:
            instruction: The instruction/path to process
            
        Returns:
            Dictionary containing contact_person, context, country, documents, and plan
        """       
        print("Planning agent asked: ", instruction)
        # Invoke agent with callbacks
        result = await self.agent.ainvoke({"messages": [HumanMessage(content=instruction)]}, config={"callbacks": [self.langfuse_handler]})
        # Extract the last AIMessage content
        #langfuse = get_client()
        #langfuse.flush()

        messages = result.get("messages", [])
        if not messages:
            return "No messages received"
        # Get the last message - handle both dict and object formats
        last_message = messages[-1]
        return last_message
