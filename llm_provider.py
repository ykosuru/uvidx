#!/usr/bin/env python3
"""
LLM Provider - Abstract class for LLM integration with search results

Provides a clean interface for sending search results to LLMs for analysis.
Handles multiple content types including text, code, and images from PDFs.
"""

import os
import base64
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path


class ContentType(Enum):
    """Types of content that can be sent to LLM"""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    MIXED = "mixed"  # Contains both text and images


@dataclass
class ContentItem:
    """A single content item for LLM input"""
    content_type: ContentType
    text: Optional[str] = None
    image_data: Optional[bytes] = None  # Raw image bytes
    image_base64: Optional[str] = None  # Base64 encoded
    image_media_type: str = "image/png"  # MIME type
    source_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_base64(self) -> Optional[str]:
        """Convert image data to base64 if not already encoded"""
        if self.image_base64:
            return self.image_base64
        if self.image_data:
            self.image_base64 = base64.b64encode(self.image_data).decode('utf-8')
            return self.image_base64
        return None
    
    def is_image(self) -> bool:
        """Check if this item contains an image"""
        return self.content_type == ContentType.IMAGE or self.image_data is not None


@dataclass
class LLMRequest:
    """Request structure for LLM invocation"""
    system_prompt: str
    user_prompt: str
    content_type: ContentType = ContentType.TEXT
    content_items: List[ContentItem] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.3
    
    def has_images(self) -> bool:
        """Check if request contains any images"""
        return any(item.is_image() for item in self.content_items)
    
    def get_text_content(self) -> str:
        """Get all text content concatenated"""
        texts = [item.text for item in self.content_items if item.text]
        return "\n\n".join(texts)
    
    def get_images(self) -> List[ContentItem]:
        """Get all image content items"""
        return [item for item in self.content_items if item.is_image()]


@dataclass
class LLMResponse:
    """Response from LLM invocation"""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[Any] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Subclass this and implement invoke_llm() for specific providers
    (OpenAI, Anthropic, Ollama, etc.)
    
    Usage:
        class MyProvider(LLMProvider):
            def invoke_llm(self, system_prompt, user_prompt, content_type, **kwargs):
                # Your implementation
                pass
        
        provider = MyProvider(model="gpt-4")
        response = provider.invoke_llm(
            system_prompt="You are an expert...",
            user_prompt="Analyze this code...",
            content_type=ContentType.CODE
        )
    """
    
    def __init__(self, 
                 model: str = "default",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs):
        """
        Initialize LLM provider.
        
        Args:
            model: Model identifier
            api_key: API key (can also use environment variables)
            base_url: Base URL for API (for self-hosted models)
            **kwargs: Additional provider-specific arguments
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs
    
    @abstractmethod
    def invoke_llm(self,
                   system_prompt: str,
                   user_prompt: str,
                   content_type: ContentType = ContentType.TEXT,
                   content_items: Optional[List[ContentItem]] = None,
                   **kwargs) -> LLMResponse:
        """
        Invoke the LLM with the given prompts and content.
        
        Args:
            system_prompt: System/instruction prompt defining LLM behavior
            user_prompt: User query/prompt
            content_type: Type of content being sent (text, code, image, mixed)
            content_items: List of content items (for multi-modal input)
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse with the model's response
            
        Note:
            Subclasses should handle:
            - Text-only requests
            - Code with syntax context
            - Images (base64 encoded)
            - Mixed content (text + images)
        """
        pass
    
    def prepare_image_content(self, content_items: List[ContentItem]) -> List[Dict[str, Any]]:
        """
        Prepare image content for API submission.
        
        Converts images to base64 and formats for common API structures.
        
        Args:
            content_items: List of ContentItem objects
            
        Returns:
            List of formatted content blocks
        """
        blocks = []
        
        for item in content_items:
            if item.content_type == ContentType.IMAGE or item.is_image():
                # Ensure base64 encoding
                b64_data = item.to_base64()
                if b64_data:
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.image_media_type,
                            "data": b64_data
                        }
                    })
            elif item.text:
                blocks.append({
                    "type": "text",
                    "text": item.text
                })
        
        return blocks
    
    def encode_image_file(self, 
                          file_path: str, 
                          media_type: Optional[str] = None) -> ContentItem:
        """
        Load and encode an image file.
        
        Args:
            file_path: Path to image file
            media_type: MIME type (auto-detected if not provided)
            
        Returns:
            ContentItem with encoded image
        """
        path = Path(file_path)
        
        # Auto-detect media type
        if not media_type:
            ext = path.suffix.lower()
            media_types = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'
            }
            media_type = media_types.get(ext, 'image/png')
        
        with open(path, 'rb') as f:
            image_data = f.read()
        
        return ContentItem(
            content_type=ContentType.IMAGE,
            image_data=image_data,
            image_media_type=media_type,
            source_file=str(path)
        )
    
    def sanitize_text_for_llm(self, text: str) -> str:
        """
        Sanitize text content before sending to LLM.
        
        Handles:
        - Invalid UTF-8 characters
        - Control characters
        - Excessive whitespace
        
        Args:
            text: Raw text content
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Handle encoding issues - replace invalid chars
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        
        # Remove null bytes and other problematic control chars
        # Keep newlines, tabs, and carriage returns
        sanitized = ''.join(
            char if char.isprintable() or char in '\n\t\r' else ' '
            for char in text
        )
        
        # Collapse multiple blank lines
        import re
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        
        return sanitized.strip()


class StubLLMProvider(LLMProvider):
    """
    Stub implementation of LLM provider for testing.
    
    Returns a formatted response showing what would be sent to the LLM.
    Replace this with actual provider implementations (OpenAI, Anthropic, etc.)
    """
    
    def __init__(self, model: str = "stub-model", **kwargs):
        super().__init__(model=model, **kwargs)
    
    def invoke_llm(self,
                   system_prompt: str,
                   user_prompt: str,
                   content_type: ContentType = ContentType.TEXT,
                   content_items: Optional[List[ContentItem]] = None,
                   **kwargs) -> LLMResponse:
        """
        Stub implementation - returns info about the request.
        
        TODO: Replace with actual API call to your LLM provider.
        """
        content_items = content_items or []
        
        # Count content types
        text_count = sum(1 for c in content_items if c.text and not c.is_image())
        image_count = sum(1 for c in content_items if c.is_image())
        
        # Build stub response
        response_lines = [
            "=" * 60,
            "LLM PROVIDER STUB - REPLACE WITH ACTUAL IMPLEMENTATION",
            "=" * 60,
            "",
            f"Provider: {self.__class__.__name__}",
            f"Model: {self.model}",
            f"Content Type: {content_type.value}",
            "",
            "--- SYSTEM PROMPT ---",
            system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
            "",
            "--- USER PROMPT ---",
            user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt,
            "",
            "--- CONTENT ITEMS ---",
            f"Text items: {text_count}",
            f"Image items: {image_count}",
            "",
            "=" * 60,
            "",
            "To implement actual LLM integration:",
            "1. Subclass LLMProvider",
            "2. Implement invoke_llm() with your API calls",
            "3. See OpenAIProvider or AnthropicProvider examples below",
            "",
            "Environment variables to set:",
            "  OPENAI_API_KEY - for OpenAI",
            "  ANTHROPIC_API_KEY - for Anthropic",
        ]
        
        return LLMResponse(
            content="\n".join(response_lines),
            model=self.model,
            provider="stub",
            success=True
        )


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider implementation.
    
    Supports GPT-4, GPT-4-Vision, GPT-3.5-turbo, etc.
    """
    
    def __init__(self, 
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    def invoke_llm(self,
                   system_prompt: str,
                   user_prompt: str,
                   content_type: ContentType = ContentType.TEXT,
                   content_items: Optional[List[ContentItem]] = None,
                   **kwargs) -> LLMResponse:
        """Invoke OpenAI API"""
        
        if not self.api_key:
            return LLMResponse(
                content="",
                model=self.model,
                provider="openai",
                success=False,
                error="OPENAI_API_KEY not set"
            )
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            # Build messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Handle multi-modal content
            if content_items and any(c.is_image() for c in content_items):
                # Vision model - build content array
                user_content = []
                user_content.append({"type": "text", "text": user_prompt})
                
                for item in content_items:
                    if item.is_image():
                        b64 = item.to_base64()
                        if b64:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{item.image_media_type};base64,{b64}"
                                }
                            })
                    elif item.text:
                        user_content.append({"type": "text", "text": item.text})
                
                messages.append({"role": "user", "content": user_content})
                
                # Use vision model if images present
                model = self.model if "vision" in self.model else "gpt-4o"
            else:
                # Text only
                full_prompt = user_prompt
                if content_items:
                    text_content = "\n\n".join(
                        self.sanitize_text_for_llm(c.text) 
                        for c in content_items if c.text
                    )
                    if text_content:
                        full_prompt = f"{user_prompt}\n\n{text_content}"
                
                messages.append({"role": "user", "content": full_prompt})
                model = self.model
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 4096)
            )
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=model,
                provider="openai",
                tokens_used=response.usage.total_tokens if response.usage else 0,
                success=True,
                raw_response=response
            )
            
        except ImportError:
            return LLMResponse(
                content="",
                model=self.model,
                provider="openai",
                success=False,
                error="openai package not installed. Run: pip install openai"
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                provider="openai",
                success=False,
                error=str(e)
            )


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider implementation.
    
    Supports Claude 3 Opus, Sonnet, Haiku with vision capabilities.
    """
    
    def __init__(self,
                 model: str = "claude-sonnet-4-20250514",
                 api_key: Optional[str] = None,
                 **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    
    def invoke_llm(self,
                   system_prompt: str,
                   user_prompt: str,
                   content_type: ContentType = ContentType.TEXT,
                   content_items: Optional[List[ContentItem]] = None,
                   **kwargs) -> LLMResponse:
        """Invoke Anthropic API"""
        
        if not self.api_key:
            return LLMResponse(
                content="",
                model=self.model,
                provider="anthropic",
                success=False,
                error="ANTHROPIC_API_KEY not set"
            )
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Build content array
            content = []
            
            # Add user prompt text
            content.append({"type": "text", "text": user_prompt})
            
            # Add content items
            if content_items:
                for item in content_items:
                    if item.is_image():
                        b64 = item.to_base64()
                        if b64:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.image_media_type,
                                    "data": b64
                                }
                            })
                    elif item.text:
                        sanitized = self.sanitize_text_for_llm(item.text)
                        if sanitized:
                            content.append({"type": "text", "text": sanitized})
            
            response = client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 4096),
                system=system_prompt,
                messages=[{"role": "user", "content": content}]
            )
            
            response_text = ""
            if response.content:
                response_text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            
            tokens = 0
            if response.usage:
                tokens = response.usage.input_tokens + response.usage.output_tokens
            
            return LLMResponse(
                content=response_text,
                model=self.model,
                provider="anthropic",
                tokens_used=tokens,
                success=True,
                raw_response=response
            )
            
        except ImportError:
            return LLMResponse(
                content="",
                model=self.model,
                provider="anthropic",
                success=False,
                error="anthropic package not installed. Run: pip install anthropic"
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                provider="anthropic",
                success=False,
                error=str(e)
            )


class OllamaProvider(LLMProvider):
    """
    Ollama local model provider.
    
    For running local models like Llama, Mistral, etc.
    """
    
    def __init__(self,
                 model: str = "llama3",
                 base_url: str = "http://localhost:11434",
                 **kwargs):
        super().__init__(model=model, base_url=base_url, **kwargs)
    
    def invoke_llm(self,
                   system_prompt: str,
                   user_prompt: str,
                   content_type: ContentType = ContentType.TEXT,
                   content_items: Optional[List[ContentItem]] = None,
                   **kwargs) -> LLMResponse:
        """Invoke Ollama API"""
        
        try:
            import requests
            
            # Build full prompt
            full_prompt = user_prompt
            if content_items:
                text_content = "\n\n".join(
                    self.sanitize_text_for_llm(c.text)
                    for c in content_items if c.text
                )
                if text_content:
                    full_prompt = f"{user_prompt}\n\n{text_content}"
            
            # Check for images - Ollama has limited vision support
            images = []
            if content_items:
                for item in content_items:
                    if item.is_image():
                        b64 = item.to_base64()
                        if b64:
                            images.append(b64)
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                "stream": False
            }
            
            # Add images if present and model supports it
            if images:
                payload["messages"][-1]["images"] = images
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=self.model,
                provider="ollama",
                tokens_used=data.get("eval_count", 0),
                success=True,
                raw_response=data
            )
            
        except ImportError:
            return LLMResponse(
                content="",
                model=self.model,
                provider="ollama",
                success=False,
                error="requests package not installed. Run: pip install requests"
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                provider="ollama",
                success=False,
                error=str(e)
            )


# =============================================================================
# Wire Payments Expert System Prompt
# =============================================================================

WIRE_PAYMENTS_SYSTEM_PROMPT = """You are an expert in wire payment systems, SWIFT messaging, and financial transaction processing. You have deep knowledge of:

- SWIFT MT messages (MT-103, MT-202, MT-199, MT-940, MT-950)
- ISO 20022 messages (pacs.008, pacs.009, pacs.004, pain.001, camt.053)
- Payment networks (Fedwire, CHIPS, ACH, SEPA)
- Compliance requirements (OFAC sanctions screening, AML/KYC, BSA)
- Payment processing workflows (validation, routing, settlement, reconciliation)
- Exception handling (payment returns, repairs, investigations)
- Legacy systems including TAL (Transaction Application Language) on HP NonStop/Tandem

Your task is to analyze the provided code and documentation artifacts to answer the user's query. 

When analyzing code:
- Identify the business logic and processing flow
- Extract validation rules and edit checks
- Note error handling and exception conditions
- Identify integration points with other systems
- Map procedures to business capabilities
- Explain TAL syntax and constructs if present

When analyzing documentation:
- Extract requirements and specifications
- Identify compliance and regulatory requirements
- Note processing rules and cutoff times
- Extract field mappings and data transformations

When analyzing images (diagrams, flowcharts, screenshots):
- Describe the visual content
- Extract any text or labels
- Interpret process flows or architectures
- Relate diagrams to code or documentation

Provide a comprehensive analysis that includes:
1. Direct answer to the user's query
2. Implementation details found in the code
3. Business rules and requirements from documentation
4. Data flow and processing sequence
5. Error conditions and exception handling
6. Compliance and regulatory considerations
7. Recommendations or observations

Be specific and cite the source (code file/procedure or document section) when referencing information."""


# =============================================================================
# Helper Functions
# =============================================================================

def create_provider(provider_name: str = "anthropic",
                    model: Optional[str] = None,
                    **kwargs) -> LLMProvider:
    """
    Factory function to create LLM provider.
    
    Args:
        provider_name: Provider name ("anthropic", "openai", "ollama", "stub")
        model: Model name (uses provider default if not specified)
        **kwargs: Additional provider arguments
        
    Returns:
        LLMProvider instance
    """
    provider_name = provider_name.lower()
    
    if provider_name == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514", **kwargs)
    elif provider_name == "openai":
        return OpenAIProvider(model=model or "gpt-4", **kwargs)
    elif provider_name == "ollama":
        return OllamaProvider(model=model or "llama3", **kwargs)
    elif provider_name == "stub":
        return StubLLMProvider(model=model or "stub-model", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def format_search_results_for_llm(results: List[Any],
                                   min_score: float = 0.70,
                                   max_chunks: int = 20,
                                   include_images: bool = True) -> tuple:
    """
    Format search results for LLM input.
    
    Args:
        results: List of SearchResult objects
        min_score: Minimum score threshold
        max_chunks: Maximum number of chunks to include
        include_images: Whether to include image content items
        
    Returns:
        Tuple of (formatted_text, list of ContentItem for images)
    """
    # Filter by score
    filtered = [r for r in results if r.combined_score >= min_score]
    
    if not filtered:
        return f"No results found with score >= {min_score:.2f}", []
    
    # Limit chunks
    filtered = filtered[:max_chunks]
    
    # Separate by type
    code_chunks = []
    doc_chunks = []
    log_chunks = []
    image_items = []
    
    for r in filtered:
        chunk = r.chunk
        source_type = chunk.source_type.value if hasattr(chunk.source_type, 'value') else str(chunk.source_type)
        
        if source_type == "code":
            code_chunks.append(r)
        elif source_type == "document":
            doc_chunks.append(r)
            
            # Check for embedded images in document metadata
            if include_images and chunk.metadata:
                images = chunk.metadata.get('images', [])
                for img in images:
                    if isinstance(img, dict) and 'data' in img:
                        image_items.append(ContentItem(
                            content_type=ContentType.IMAGE,
                            image_data=img.get('data') if isinstance(img.get('data'), bytes) else None,
                            image_base64=img.get('data') if isinstance(img.get('data'), str) else None,
                            image_media_type=img.get('media_type', 'image/png'),
                            source_file=chunk.source_ref.file_path,
                            metadata={'page': chunk.source_ref.page_number}
                        ))
        elif source_type == "log":
            log_chunks.append(r)
    
    # Build formatted text output
    sections = []
    
    # Code section
    if code_chunks:
        sections.append("=" * 60)
        sections.append("CODE ARTIFACTS")
        sections.append("=" * 60)
        
        for r in code_chunks:
            chunk = r.chunk
            ref = chunk.source_ref
            
            sections.append("")
            sections.append(f"--- File: {ref.file_path or 'unknown'} ---")
            if ref.procedure_name:
                sections.append(f"Procedure: {ref.procedure_name}")
            if ref.line_start:
                line_info = f"Lines: {ref.line_start}"
                if ref.line_end:
                    line_info += f"-{ref.line_end}"
                sections.append(line_info)
            sections.append(f"Score: {r.combined_score:.3f}")
            
            if r.matched_concepts:
                sections.append(f"Concepts: {', '.join(r.matched_concepts[:5])}")
            
            caps = list(chunk.capability_set)[:5]
            if caps:
                sections.append(f"Capabilities: {', '.join(caps)}")
            
            sections.append("")
            sections.append("```")
            sections.append(chunk.text.strip())
            sections.append("```")
    
    # Document section
    if doc_chunks:
        sections.append("")
        sections.append("=" * 60)
        sections.append("DOCUMENTATION ARTIFACTS")
        sections.append("=" * 60)
        
        for r in doc_chunks:
            chunk = r.chunk
            ref = chunk.source_ref
            
            sections.append("")
            sections.append(f"--- Document: {ref.file_path or 'unknown'} ---")
            if ref.page_number:
                sections.append(f"Page: {ref.page_number}")
            if ref.section_title:
                sections.append(f"Section: {ref.section_title}")
            sections.append(f"Score: {r.combined_score:.3f}")
            
            if r.matched_concepts:
                sections.append(f"Concepts: {', '.join(r.matched_concepts[:5])}")
            
            sections.append("")
            sections.append(chunk.text.strip())
    
    # Log section
    if log_chunks:
        sections.append("")
        sections.append("=" * 60)
        sections.append("LOG ARTIFACTS")
        sections.append("=" * 60)
        
        for r in log_chunks:
            chunk = r.chunk
            ref = chunk.source_ref
            
            sections.append("")
            sections.append(f"--- Log: {ref.file_path or 'unknown'} ---")
            sections.append(f"Score: {r.combined_score:.3f}")
            sections.append("")
            sections.append(chunk.text.strip())
    
    return "\n".join(sections), image_items


def analyze_search_results(query: str,
                           results: List[Any],
                           provider: LLMProvider,
                           min_score: float = 0.70,
                           max_chunks: int = 20,
                           system_prompt: Optional[str] = None,
                           verbose: bool = False) -> LLMResponse:
    """
    Analyze search results using LLM.
    
    Args:
        query: User's search query
        results: List of SearchResult objects
        provider: LLMProvider instance
        min_score: Minimum score threshold for including results
        max_chunks: Maximum number of chunks to include
        system_prompt: Custom system prompt (uses default if None)
        verbose: Print debug information
        
    Returns:
        LLMResponse with analysis
    """
    # Format results
    context_text, image_items = format_search_results_for_llm(
        results, min_score, max_chunks, include_images=True
    )
    
    if verbose:
        print(f"\n📄 Context length: {len(context_text)} characters")
        print(f"📊 Results with score >= {min_score}: {len([r for r in results if r.combined_score >= min_score])}")
        print(f"🖼️  Images found: {len(image_items)}")
    
    # Check for empty results
    if "No results found" in context_text:
        return LLMResponse(
            content=f"No search results found with score >= {min_score}. Try lowering the threshold or broadening your query.",
            model=provider.model,
            provider="none",
            success=False,
            error="No results to analyze"
        )
    
    # Build prompts
    sys_prompt = system_prompt or WIRE_PAYMENTS_SYSTEM_PROMPT
    
    user_prompt = f"""USER QUERY:
{query}

RELEVANT ARTIFACTS:
{context_text}

Please analyze the above artifacts and provide a comprehensive answer to the user's query. 
Extract implementation details, business rules, and any relevant technical information.
If there are images referenced, describe what you see in them."""
    
    # Determine content type
    content_type = ContentType.MIXED if image_items else ContentType.TEXT
    
    # Build content items
    content_items = [ContentItem(content_type=ContentType.TEXT, text=context_text)]
    content_items.extend(image_items)
    
    if verbose:
        print(f"🤖 Invoking {provider.__class__.__name__} ({provider.model})...")
    
    # Invoke LLM
    response = provider.invoke_llm(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        content_type=content_type,
        content_items=content_items
    )
    
    if verbose and response.tokens_used:
        print(f"📈 Tokens used: {response.tokens_used}")
    
    return response


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core classes
    'LLMProvider',
    'LLMRequest',
    'LLMResponse',
    'ContentType',
    'ContentItem',
    
    # Providers
    'StubLLMProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'OllamaProvider',
    
    # Factory and helpers
    'create_provider',
    'format_search_results_for_llm',
    'analyze_search_results',
    
    # Constants
    'WIRE_PAYMENTS_SYSTEM_PROMPT',
]
