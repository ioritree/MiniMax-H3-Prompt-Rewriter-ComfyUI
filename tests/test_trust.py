"""Downloads through an antivirus that inspects HTTPS.

An antivirus that scans encrypted connections re-signs huggingface.co with a root
it installs in the Windows store. The browser trusts that store; ``requests`` trusts
certifi's bundle and nothing else, so every download failed with "self-signed
certificate in certificate chain" while the same file opened fine in a browser.

Nothing here touches the network. What is pinned is the wiring: every request
the pack makes goes through an adapter whose TLS context is the system's, and
a certificate that still fails says what is in the way.
"""

import importlib
import pathlib
import ssl
import sys
import types

import requests

_PKG = "minimax_h3_rewriter"
ROOT = pathlib.Path(__file__).resolve().parent.parent

if _PKG not in sys.modules:
    _package = types.ModuleType(_PKG)
    _package.__path__ = [str(ROOT / _PKG)]
    sys.modules[_PKG] = _package

download = importlib.import_module(f"{_PKG}.download")
hub_sync = importlib.import_module(f"{_PKG}.hub_sync")


def test_the_adapter_hands_urllib3_the_system_context():
    adapter = download._SystemTrust()
    context = adapter.poolmanager.connection_pool_kw.get("ssl_context")
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


def test_a_proxy_gets_the_same_context():
    adapter = download._SystemTrust()
    manager = adapter.proxy_manager_for("http://proxy.example:3128")
    assert manager.connection_pool_kw.get("ssl_context") is download._SystemTrust.context()


def test_the_context_holds_the_system_store():
    """On Windows ``create_default_context`` reads the ROOT and CA stores."""
    context = download._SystemTrust.context()
    if sys.platform == "win32":
        assert context.cert_store_stats()["x509_ca"] > 0


def test_every_request_goes_through_the_adapter(monkeypatch):
    seen = {}

    def send(self, request, **kwargs):
        seen["adapter"] = self
        response = requests.Response()
        response.status_code = 200
        response.url = request.url
        return response

    monkeypatch.setattr(download._SystemTrust, "send", send)
    download.http("GET", "https://huggingface.co/api/models/x")
    assert isinstance(seen["adapter"], download._SystemTrust)


def test_no_module_calls_requests_directly():
    """A new ``requests.get`` would go round the system store again."""
    offenders = []
    for path in sorted((ROOT / _PKG).glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for call in ("requests.get(", "requests.head(", "requests.post(", "requests.request("):
            if call in text:
                offenders.append(f"{path.name}: {call}")
    assert not offenders, offenders


def test_a_certificate_that_still_fails_says_what_is_in_the_way():
    error = requests.exceptions.SSLError("certificate verify failed: self-signed certificate")
    message = download.unreachable("https://huggingface.co", error)
    assert download.UNTRUSTED_HINT in message


def test_an_ordinary_network_failure_gets_no_certificate_lecture():
    error = requests.exceptions.ConnectionError("connection refused")
    assert download.UNTRUSTED_HINT not in download.unreachable("https://huggingface.co", error)


def test_huggingface_hub_errors_are_recognised_under_their_wrappers():
    inner = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    try:
        try:
            raise inner
        except ssl.SSLError as error:
            raise RuntimeError("httpx.ConnectError") from error
    except RuntimeError as wrapped:
        assert hub_sync._untrusted(wrapped)
    assert hub_sync._untrusted(RuntimeError("invalid peer certificate: UnknownIssuer"))
    assert not hub_sync._untrusted(RuntimeError("404 Not Found"))
