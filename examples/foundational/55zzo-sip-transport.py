#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SIP Transport Example - Echo Bot

Accepts incoming SIP calls and echoes received audio back to the caller.
Useful for testing SIP connectivity and verifying audio round-trip.

Requirements:
    pip install "pipecat-ai[sip]"

Testing with pjsua (CLI SIP client):
    pjsua --null-audio sip:bot@<HOST>:5060

Testing with Linphone:
    1. Open Linphone
    2. Call sip:bot@<HOST>:5060

Environment variables:
    SIP_HOST: Bind address (default: 0.0.0.0)
    SIP_PORT: SIP port (default: 5060)
"""

import asyncio
import logging
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.sip import SIPCallTransport, SIPParams, SIPServerTransport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    sip_host = os.getenv("SIP_HOST", "0.0.0.0")
    sip_port = int(os.getenv("SIP_PORT", "5060"))

    params = SIPParams(
        sip_listen_host=sip_host,
        sip_listen_port=sip_port,
        audio_in_enabled=True,
        audio_out_enabled=True,
    )

    server = SIPServerTransport(params=params)

    @server.event_handler("on_call_started")
    async def on_call(server, call_transport: SIPCallTransport):
        logger.info("Call started: %s", call_transport.session.call_id)

        # Echo pipeline: input -> output (echoes audio back)
        pipeline = Pipeline([call_transport.input(), call_transport.output()])

        runner = PipelineRunner()
        task = PipelineTask(pipeline)
        await runner.run(task)

        logger.info("Call ended: %s", call_transport.session.call_id)

    @server.event_handler("on_call_ended")
    async def on_call_ended(server, call_transport: SIPCallTransport):
        logger.info("Call ended (BYE): %s", call_transport.session.call_id)

    await server.start()
    logger.info("SIP echo bot listening on %s:%d", sip_host, sip_port)

    # Run until interrupted
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
