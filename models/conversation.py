from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class Message(BaseModel):
    """Model for a conversation message"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        frozen = False


class SQLExecution(BaseModel):
    """Model for SQL execution details"""
    query: str
    execution_time: float
    row_count: int = 0
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Conversation(BaseModel):
    """Model for a conversation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    sql_executions: List[SQLExecution] = []
    metadata: Dict[str, Any] = {}

    def add_id(self, id: str) -> str:
        self.id = id
        return id
    
    def add_message(self, role: str, content: str) -> Message:
        """Add a message to the conversation"""
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message
    
    def add_sql_execution(self, query: str, execution_time: float, 
                          row_count: int = 0, error: Optional[str] = None) -> SQLExecution:
        """Add an SQL execution record to the conversation"""
        sql_execution = SQLExecution(
            query=query,
            execution_time=execution_time,
            row_count=row_count,
            error=error
        )
        self.sql_executions.append(sql_execution)
        return sql_execution
    
    def get_message_history(self) -> List[Dict[str, str]]:
        """Get the message history in a format suitable for LLM context"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


class ConversationStore(BaseModel):
    """In-memory store for conversations"""
    conversations: Dict[str, Conversation] = {}
    
    def create_conversation(self, conversation_id: str) -> Conversation:
        """Create a new conversation and store it"""
        conversation = Conversation()
        conversation.add_id(conversation_id)
        
        self.conversations[conversation.id] = conversation
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self.conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation by ID"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
    
    def list_conversations(self) -> List[Conversation]:
        """List all conversations"""
        return list(self.conversations.values())


# Create a global in-memory conversation store
conversation_store = ConversationStore()