"""APPFL gRPC calibration demo."""

import logging
import threading
from concurrent import futures

import grpc
import numpy as np
import yaml
from omegaconf import OmegaConf

from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCClientCommunicator, GRPCServerCommunicator
from appfl.comm.grpc.grpc_communicator_pb2 import (
    CustomActionRequest,
    CustomActionResponse,
    ServerHeader,
    ServerStatus,
)
from appfl.comm.grpc.grpc_communicator_pb2_grpc import (
    add_GRPCCommunicatorServicer_to_server,
)
from appfl.comm.grpc.utils import proto_to_databuffer

from appfl_adapter import (
    ACTION_NAME,
    aggregate_action_payloads,
    make_action_payload,
)
from demo import make_demo_sites
from fedcal import compute_client_report, pooled_subgroup_metrics


class CalibrationServerAgent(ServerAgent):
    """Minimal evaluation server."""

    def __init__(self, num_clients):
        self.server_agent_config = OmegaConf.create({"server_configs": {}})
        self.num_clients = num_clients


class CalibrationServerCommunicator(GRPCServerCommunicator):
    """Handle calibration requests."""

    def __init__(self, server_agent, **kwargs):
        super().__init__(server_agent, **kwargs)
        self._reports = {}
        self._result_futures = {}
        self._reports_lock = threading.Lock()

    def _submit_report(self, client_id, payload):
        if not isinstance(payload, dict) or payload.get("client_id") != client_id:
            raise ValueError("gRPC client ID does not match report client ID")

        result_future = futures.Future()
        with self._reports_lock:
            if client_id in self._reports:
                raise ValueError(f"duplicate report from client {client_id}")
            self._reports[client_id] = payload
            self._result_futures[client_id] = result_future

            if len(self._reports) == self.server_agent.get_num_clients():
                try:
                    result = aggregate_action_payloads(list(self._reports.values()))
                except Exception as error:
                    for waiting_future in self._result_futures.values():
                        waiting_future.set_exception(error)
                else:
                    for waiting_future in self._result_futures.values():
                        waiting_future.set_result(result)
                finally:
                    self._reports = {}
                    self._result_futures = {}

        return result_future.result(timeout=30)

    def InvokeCustomAction(self, request_iterator, context):
        request = CustomActionRequest()
        request.ParseFromString(
            b"".join(chunk.data_bytes for chunk in request_iterator)
        )

        if request.action != ACTION_NAME:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"custom action {request.action!r} is not implemented",
            )

        try:
            metadata = yaml.safe_load(request.meta_data) if request.meta_data else {}
        except yaml.YAMLError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        if not isinstance(metadata, dict) or set(metadata) != {"report"}:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "the calibration action requires one report",
            )

        try:
            result = self._submit_report(
                request.header.client_id,
                metadata["report"],
            )
        except (TypeError, ValueError, TimeoutError) as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception as error:
            context.abort(grpc.StatusCode.INTERNAL, str(error))

        response = CustomActionResponse(
            header=ServerHeader(status=ServerStatus.DONE),
            results=yaml.safe_dump(result),
        )
        yield from proto_to_databuffer(
            response,
            max_message_size=self.max_message_size,
        )


def _send_report(server_uri, client_id, site):
    report = compute_client_report(client_id, *site)
    communicator = GRPCClientCommunicator(
        client_id=client_id,
        server_uri=server_uri,
        use_ssl=False,
    )
    return communicator.invoke_custom_action(
        action=ACTION_NAME,
        report=make_action_payload(report),
    )


def _pooled_result(sites):
    return pooled_subgroup_metrics(
        np.concatenate([site[0] for site in sites.values()]),
        np.concatenate([site[1] for site in sites.values()]),
        np.concatenate([site[2] for site in sites.values()]),
    )


def main():
    sites = make_demo_sites()
    server_agent = CalibrationServerAgent(num_clients=len(sites))
    communicator = CalibrationServerCommunicator(
        server_agent,
        logger=logging.getLogger("appfl-calibration-server"),
    )

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    add_GRPCCommunicatorServicer_to_server(communicator, grpc_server)
    port = grpc_server.add_insecure_port("127.0.0.1:0")
    grpc_server.start()
    server_uri = f"127.0.0.1:{port}"

    try:
        with futures.ThreadPoolExecutor(max_workers=len(sites)) as executor:
            result_futures = [
                executor.submit(_send_report, server_uri, client_id, site)
                for client_id, site in sites.items()
            ]
            client_results = [future.result(timeout=60) for future in result_futures]
    finally:
        grpc_server.stop(grace=0).wait(timeout=5)

    federated = client_results[0]
    if any(result != federated for result in client_results[1:]):
        raise RuntimeError("APPFL clients received different aggregate results")

    pooled = _pooled_result(sites)
    max_difference = max(
        max(
            abs(federated[group]["ece"] - pooled[group]["ece"]),
            abs(federated[group]["brier"] - pooled[group]["brier"]),
        )
        for group in federated
    )
    if max_difference >= 1e-12:
        raise RuntimeError("APPFL result does not match the pooled reference")

    print("APPFL gRPC custom action completed")
    print(f"action: {ACTION_NAME}")
    print(f"clients: {len(sites)}")
    for group in sorted(federated):
        print(
            f"{group}: n={federated[group]['n']}, "
            f"ECE={federated[group]['ece']:.6f}, "
            f"Brier={federated[group]['brier']:.6f}"
        )
    print(f"max |APPFL - pooled| = {max_difference:.2e}")


if __name__ == "__main__":
    main()
