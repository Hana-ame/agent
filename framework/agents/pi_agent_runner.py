import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("vertex_edge_agent.agents")
from .base_agent import BaseAgent

class PiAgentRunner(BaseAgent):
    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        import asyncio
        import json
        
        logger.info(f"[PiAgentRunner] Handing off to real Pi Agent CLI (model={model})")
        
        if isinstance(data, (dict, list)):
            message = json.dumps(data, ensure_ascii=False)
        else:
            message = str(data)

        cmd = ["pi", "-p"]
        
        if model and model != "default":
            cmd.extend(["--model", model])
            
        if prompt:
            cmd.extend(["--system-prompt", prompt])
            
        settings = settings or {}
        
        if "mode" in settings:
            cmd.extend(["--mode", str(settings["mode"])])
        if "tools" in settings:
            cmd.extend(["--tools", str(settings["tools"])])
        if "thinking" in settings:
            cmd.extend(["--thinking", str(settings["thinking"])])
        if "api_key" in settings:
            cmd.extend(["--api-key", str(settings["api_key"])])
        if "provider" in settings:
            cmd.extend(["--provider", str(settings["provider"])])
            
        cmd.append("--")
        cmd.append(message)

        logger.debug(f"[PiAgentRunner] Executing: {' '.join(cmd)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode('utf-8').strip()
                logger.error(f"[PiAgentRunner] Pi CLI failed with code {proc.returncode}: {error_msg}")
                raise RuntimeError(f"Pi Agent CLI error: {error_msg}")
                
            output = stdout.decode('utf-8').strip()
            
            if settings.get("mode") == "json":
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    logger.warning("[PiAgentRunner] Failed to parse JSON output, returning raw string.")
                    return output
                    
            return output
            
        except FileNotFoundError:
            logger.error(
                "[PiAgentRunner] 'pi' command not found. "
                "Please make sure the Pi Agent CLI is installed and in your PATH."
            )
            raise

