"""
Log Parser - Extracts indexable chunks from transaction logs

Handles JSON logs, structured logs, and pattern-based parsing
with trace correlation for grouping related log entries.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from .base import ContentParser
from ..models import (
    IndexableChunk,
    SourceType,
    SemanticType,
    SourceReference,
    DomainMatch
)
from ..vocabulary import DomainVocabulary


@dataclass
class LogEntry:
    """Represents a single log entry"""
    timestamp: Optional[str]
    level: Optional[str]  # ERROR, WARN, INFO, DEBUG
    message: str
    raw_line: str
    line_number: int
    
    # Correlation IDs
    trace_id: Optional[str] = None
    transaction_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Structured data
    fields: Dict[str, Any] = field(default_factory=dict)
    
    # Classification
    is_error: bool = False
    error_code: Optional[str] = None


@dataclass 
class LogTrace:
    """A group of related log entries (same trace/transaction)"""
    trace_id: str
    entries: List[LogEntry] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    has_error: bool = False
    error_codes: List[str] = field(default_factory=list)


class LogParser(ContentParser):
    """
    Parser for log files with payment transaction awareness
    
    Supports JSON logs, structured logs, and common log formats.
    Groups entries by trace/transaction ID for correlated analysis.
    """
    
    SOURCE_TYPE = SourceType.LOG
    SUPPORTED_EXTENSIONS = ['.log', '.json', '.jsonl', '.txt']
    
    # Common log patterns
    PATTERNS = {
        # ISO timestamp pattern
        'timestamp': r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)',
        
        # Log levels
        'level': r'\b(ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE|FATAL|CRITICAL)\b',
        
        # Transaction/Trace IDs (common formats)
        'trace_id': r'(?:trace[_-]?id|x-trace-id|traceid)[=:\s]+([a-zA-Z0-9\-_]{8,})',
        'transaction_id': r'(?:transaction[_-]?id|txn[_-]?id|trxid)[=:\s]+([a-zA-Z0-9\-_]{6,})',
        'request_id': r'(?:request[_-]?id|req[_-]?id)[=:\s]+([a-zA-Z0-9\-_]{6,})',
        
        # Payment-specific patterns
        'imad': r'\b(IMAD)[:\s]+([A-Z0-9]{20,})',
        'omad': r'\b(OMAD)[:\s]+([A-Z0-9]{20,})',
        'swift_msg': r'\b(MT\d{3})\b',
        'error_code': r'(?:error[_-]?code|err[_-]?code)[=:\s]+([A-Z0-9_\-]+)',
        
        # Amount patterns
        'amount': r'(?:amount|amt)[=:\s]+(\d+(?:\.\d{2})?)',
        'currency': r'(?:currency|ccy)[=:\s]+([A-Z]{3})',
    }
    
    def __init__(self, 
                 vocabulary: DomainVocabulary,
                 group_by_trace: bool = True,
                 max_trace_entries: int = 50):
        """
        Initialize log parser
        
        Args:
            vocabulary: Domain vocabulary for concept matching
            group_by_trace: Whether to group entries by trace ID
            max_trace_entries: Max entries to group per trace
        """
        super().__init__(vocabulary)
        self.group_by_trace = group_by_trace
        self.max_trace_entries = max_trace_entries
        
        # Compile patterns
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }
    
    def can_parse(self, file_path: str) -> bool:
        """Check if file is a log file"""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def parse(self, content: bytes, source_path: str) -> List[IndexableChunk]:
        """Parse log content and extract chunks"""
        text = content.decode('utf-8', errors='replace')
        
        # Detect format and parse
        if self._is_json_lines(text):
            entries = self._parse_json_lines(text, source_path)
        elif self._is_json_array(text):
            entries = self._parse_json_array(text, source_path)
        else:
            entries = self._parse_text_logs(text, source_path)
        
        # Group by trace if enabled
        if self.group_by_trace:
            traces = self._group_by_trace(entries)
            chunks = self._traces_to_chunks(traces, source_path)
        else:
            chunks = self._entries_to_chunks(entries, source_path)
        
        return chunks
    
    def _is_json_lines(self, text: str) -> bool:
        """Check if text is JSON Lines format"""
        lines = text.strip().split('\n')[:5]
        if not lines:
            return False
        
        json_count = 0
        for line in lines:
            line = line.strip()
            if line and line.startswith('{') and line.endswith('}'):
                try:
                    json.loads(line)
                    json_count += 1
                except:
                    pass
        
        return json_count >= len(lines) // 2
    
    def _is_json_array(self, text: str) -> bool:
        """Check if text is a JSON array"""
        text = text.strip()
        if text.startswith('[') and text.endswith(']'):
            try:
                data = json.loads(text)
                return isinstance(data, list)
            except:
                pass
        return False
    
    def _parse_json_lines(self, text: str, source_path: str) -> List[LogEntry]:
        """Parse JSON Lines format"""
        entries = []
        
        for line_num, line in enumerate(text.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                entry = self._json_to_log_entry(data, line, line_num)
                entries.append(entry)
            except json.JSONDecodeError:
                # Fall back to text parsing
                entry = self._parse_text_line(line, line_num)
                entries.append(entry)
        
        return entries
    
    def _parse_json_array(self, text: str, source_path: str) -> List[LogEntry]:
        """Parse JSON array format"""
        entries = []
        
        try:
            data = json.loads(text)
            for i, item in enumerate(data, 1):
                entry = self._json_to_log_entry(item, json.dumps(item), i)
                entries.append(entry)
        except json.JSONDecodeError:
            # Fall back to text parsing
            return self._parse_text_logs(text, source_path)
        
        return entries
    
    def _json_to_log_entry(self, data: Dict, raw_line: str, line_num: int) -> LogEntry:
        """Convert JSON object to LogEntry"""
        # Extract common fields (handle various naming conventions)
        timestamp = (
            data.get('timestamp') or 
            data.get('time') or 
            data.get('@timestamp') or
            data.get('datetime') or
            data.get('ts')
        )
        
        level = (
            data.get('level') or 
            data.get('severity') or 
            data.get('log_level') or
            data.get('loglevel')
        )
        if level:
            level = str(level).upper()
        
        message = (
            data.get('message') or 
            data.get('msg') or 
            data.get('log') or
            data.get('text') or
            str(data)
        )
        
        # Extract correlation IDs
        trace_id = (
            data.get('trace_id') or 
            data.get('traceId') or 
            data.get('x-trace-id') or
            data.get('spanId')
        )
        
        transaction_id = (
            data.get('transaction_id') or 
            data.get('transactionId') or 
            data.get('txn_id') or
            data.get('txnId') or
            data.get('payment_id') or
            data.get('paymentId')
        )
        
        request_id = (
            data.get('request_id') or 
            data.get('requestId') or 
            data.get('req_id')
        )
        
        # Detect errors
        is_error = level in ['ERROR', 'FATAL', 'CRITICAL'] if level else False
        error_code = data.get('error_code') or data.get('errorCode')
        
        return LogEntry(
            timestamp=str(timestamp) if timestamp else None,
            level=level,
            message=str(message),
            raw_line=raw_line,
            line_number=line_num,
            trace_id=str(trace_id) if trace_id else None,
            transaction_id=str(transaction_id) if transaction_id else None,
            request_id=str(request_id) if request_id else None,
            fields=data,
            is_error=is_error,
            error_code=str(error_code) if error_code else None
        )
    
    def _parse_text_logs(self, text: str, source_path: str) -> List[LogEntry]:
        """Parse text-based log files"""
        entries = []
        
        for line_num, line in enumerate(text.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            
            entry = self._parse_text_line(line, line_num)
            entries.append(entry)
        
        return entries
    
    def _parse_text_line(self, line: str, line_num: int) -> LogEntry:
        """Parse a single text log line"""
        # Extract timestamp
        timestamp = None
        ts_match = self._compiled_patterns['timestamp'].search(line)
        if ts_match:
            timestamp = ts_match.group(1)
        
        # Extract level
        level = None
        level_match = self._compiled_patterns['level'].search(line)
        if level_match:
            level = level_match.group(1).upper()
        
        # Extract correlation IDs
        trace_id = None
        trace_match = self._compiled_patterns['trace_id'].search(line)
        if trace_match:
            trace_id = trace_match.group(1)
        
        transaction_id = None
        txn_match = self._compiled_patterns['transaction_id'].search(line)
        if txn_match:
            transaction_id = txn_match.group(1)
        
        request_id = None
        req_match = self._compiled_patterns['request_id'].search(line)
        if req_match:
            request_id = req_match.group(1)
        
        # Extract error code
        error_code = None
        err_match = self._compiled_patterns['error_code'].search(line)
        if err_match:
            error_code = err_match.group(1)
        
        # Extract payment-specific fields
        fields = {}
        
        imad_match = self._compiled_patterns['imad'].search(line)
        if imad_match:
            fields['imad'] = imad_match.group(2)
        
        omad_match = self._compiled_patterns['omad'].search(line)
        if omad_match:
            fields['omad'] = omad_match.group(2)
        
        swift_match = self._compiled_patterns['swift_msg'].search(line)
        if swift_match:
            fields['message_type'] = swift_match.group(1)
        
        amount_match = self._compiled_patterns['amount'].search(line)
        if amount_match:
            fields['amount'] = amount_match.group(1)
        
        currency_match = self._compiled_patterns['currency'].search(line)
        if currency_match:
            fields['currency'] = currency_match.group(1)
        
        # Detect errors
        is_error = level in ['ERROR', 'FATAL', 'CRITICAL'] if level else 'error' in line.lower()
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            message=line,
            raw_line=line,
            line_number=line_num,
            trace_id=trace_id,
            transaction_id=transaction_id,
            request_id=request_id,
            fields=fields,
            is_error=is_error,
            error_code=error_code
        )
    
    def _group_by_trace(self, entries: List[LogEntry]) -> List[LogTrace]:
        """Group log entries by trace/transaction ID"""
        traces_by_id: Dict[str, LogTrace] = {}
        orphan_entries: List[LogEntry] = []
        
        for entry in entries:
            # Determine grouping ID (prefer trace_id, then transaction_id)
            group_id = entry.trace_id or entry.transaction_id or entry.request_id
            
            if group_id:
                if group_id not in traces_by_id:
                    traces_by_id[group_id] = LogTrace(trace_id=group_id)
                
                trace = traces_by_id[group_id]
                if len(trace.entries) < self.max_trace_entries:
                    trace.entries.append(entry)
                    
                    # Update trace metadata
                    if not trace.start_time and entry.timestamp:
                        trace.start_time = entry.timestamp
                    if entry.timestamp:
                        trace.end_time = entry.timestamp
                    
                    if entry.is_error:
                        trace.has_error = True
                    if entry.error_code and entry.error_code not in trace.error_codes:
                        trace.error_codes.append(entry.error_code)
            else:
                orphan_entries.append(entry)
        
        # Create traces from orphan entries (group by time proximity or individually)
        traces = list(traces_by_id.values())
        
        # Add orphan entries as individual "traces"
        for entry in orphan_entries:
            trace = LogTrace(
                trace_id=f"orphan_{entry.line_number}",
                entries=[entry],
                start_time=entry.timestamp,
                end_time=entry.timestamp,
                has_error=entry.is_error,
                error_codes=[entry.error_code] if entry.error_code else []
            )
            traces.append(trace)
        
        return traces
    
    def _traces_to_chunks(self, traces: List[LogTrace], source_path: str) -> List[IndexableChunk]:
        """Convert log traces to indexable chunks"""
        chunks = []
        
        for trace in traces:
            if not trace.entries:
                continue
            
            # Combine all entry messages
            combined_text = '\n'.join(e.message for e in trace.entries)
            
            # Match domain concepts
            domain_matches = self.match_domain_concepts(combined_text)
            
            # Determine semantic type
            if trace.has_error:
                semantic_type = SemanticType.ERROR_TRACE
            else:
                semantic_type = SemanticType.TRANSACTION
            
            # Build metadata
            metadata = {
                'entry_count': len(trace.entries),
                'has_error': trace.has_error
            }
            
            if trace.error_codes:
                metadata['error_codes'] = trace.error_codes
            
            # Collect all unique fields
            all_fields = {}
            for entry in trace.entries:
                all_fields.update(entry.fields)
            
            if all_fields:
                metadata['fields'] = all_fields
            
            # Get first entry for line reference
            first_entry = trace.entries[0]
            last_entry = trace.entries[-1]
            
            # Create source reference
            source_ref = SourceReference(
                file_path=source_path,
                line_start=first_entry.line_number,
                line_end=last_entry.line_number,
                timestamp=trace.start_time,
                trace_id=trace.trace_id if not trace.trace_id.startswith('orphan_') else None,
                transaction_id=first_entry.transaction_id
            )
            
            # Create embedding text
            embedding_text = self.create_embedding_text(
                combined_text,
                semantic_type,
                domain_matches,
                metadata
            )
            
            chunk = IndexableChunk(
                chunk_id=self.generate_chunk_id(source_path, combined_text, trace.trace_id),
                text=combined_text,
                embedding_text=embedding_text,
                source_type=SourceType.LOG,
                semantic_type=semantic_type,
                source_ref=source_ref,
                domain_matches=domain_matches,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _entries_to_chunks(self, entries: List[LogEntry], source_path: str) -> List[IndexableChunk]:
        """Convert individual log entries to chunks (no grouping)"""
        chunks = []
        
        # Group nearby entries to avoid too many tiny chunks
        batch_size = 10
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            
            combined_text = '\n'.join(e.message for e in batch)
            
            domain_matches = self.match_domain_concepts(combined_text)
            
            # Check for errors in batch
            has_error = any(e.is_error for e in batch)
            semantic_type = SemanticType.ERROR_TRACE if has_error else SemanticType.AUDIT_ENTRY
            
            metadata = {
                'entry_count': len(batch),
                'has_error': has_error,
                'levels': list(set(e.level for e in batch if e.level))
            }
            
            first_entry = batch[0]
            last_entry = batch[-1]
            
            source_ref = SourceReference(
                file_path=source_path,
                line_start=first_entry.line_number,
                line_end=last_entry.line_number,
                timestamp=first_entry.timestamp
            )
            
            embedding_text = self.create_embedding_text(
                combined_text,
                semantic_type,
                domain_matches,
                metadata
            )
            
            chunk = IndexableChunk(
                chunk_id=self.generate_chunk_id(source_path, combined_text, str(i)),
                text=combined_text,
                embedding_text=embedding_text,
                source_type=SourceType.LOG,
                semantic_type=semantic_type,
                source_ref=source_ref,
                domain_matches=domain_matches,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def parse_streaming(self, 
                        file_path: str,
                        batch_size: int = 1000) -> Generator[List[IndexableChunk], None, None]:
        """
        Parse large log files in streaming fashion
        
        Yields batches of chunks to handle very large files.
        
        Args:
            file_path: Path to log file
            batch_size: Number of entries per batch
            
        Yields:
            Lists of IndexableChunk objects
        """
        entries = []
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Try JSON first
                try:
                    data = json.loads(line)
                    entry = self._json_to_log_entry(data, line, line_num)
                except:
                    entry = self._parse_text_line(line, line_num)
                
                entries.append(entry)
                
                if len(entries) >= batch_size:
                    if self.group_by_trace:
                        traces = self._group_by_trace(entries)
                        chunks = self._traces_to_chunks(traces, file_path)
                    else:
                        chunks = self._entries_to_chunks(entries, file_path)
                    
                    yield chunks
                    entries = []
        
        # Process remaining entries
        if entries:
            if self.group_by_trace:
                traces = self._group_by_trace(entries)
                chunks = self._traces_to_chunks(traces, file_path)
            else:
                chunks = self._entries_to_chunks(entries, file_path)
            
            yield chunks
