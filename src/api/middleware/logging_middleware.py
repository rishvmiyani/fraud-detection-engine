from fastapi import Request
import logging

logger = logging.getLogger(__name__)

class LoggingMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        # TODO: Implement logging middleware
        await self.app(scope, receive, send)
