from typing import Optional
import os

try:
    import google.genai as genai  # type: ignore
    GENAI = genai
except Exception:
    try:
        import google.generativeai as genai  # type: ignore
        GENAI = genai
    except Exception:
        GENAI = None


def configure_genai(api_key: str) -> None:
    """Configure the available genai module for API usage.

    This function tries several strategies to be compatible with different
    versions of the Google generative AI libraries.
    """
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required")

    if GENAI is None:
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        return

    # 1) Preferred: genai.configure(api_key=...)
    configure_fn = getattr(GENAI, "configure", None)
    if callable(configure_fn):
        try:
            configure_fn(api_key=api_key)
            return
        except Exception:
            # Fallthrough to other methods
            pass

    # 2) Newer API surface may provide a Client class
    ClientCls = getattr(GENAI, "Client", None)
    if ClientCls:
        try:
            client = ClientCls(api_key=api_key)
            # attach for other modules to reuse if they need
            setattr(GENAI, "_client", client)
            return
        except Exception:
            pass

    # 3) Some lightweight libraries accept setting an attribute
    try:
        setattr(GENAI, "api_key", api_key)
        return
    except Exception:
        pass

    # 4) Last resort: set env var so underlying HTTP client can pick it up
    os.environ.setdefault("GOOGLE_API_KEY", api_key)


class _CompatResponse:
    def __init__(self, text: str):
        self.text = text


def get_model(model_name: str, generation_config: Optional[object] = None):
    """Return a model-like object with async `generate_content_async(prompt)`.

    The returned object will try to call the underlying library's generation
    function in a best-effort manner. If no supported client is available an
    informative RuntimeError is raised.
    """
    if GENAI is None:
        raise RuntimeError("No Google generative AI library available")

    # Preferred: library exposes GenerativeModel class
    GenerativeModelCls = getattr(GENAI, "GenerativeModel", None)
    if GenerativeModelCls:
        try:
            model = GenerativeModelCls(model_name, generation_config=generation_config) if generation_config is not None else GenerativeModelCls(model_name)
            return model
        except Exception:
            # Fall through to other strategies
            pass

    # If a client instance was attached during configure_genai
    client = getattr(GENAI, "_client", None)
    if client is None:
        # Try to instantiate Client if available
        ClientCls = getattr(GENAI, "Client", None)
        if ClientCls:
            try:
                client = ClientCls(api_key=os.environ.get("GOOGLE_API_KEY"))
                setattr(GENAI, "_client", client)
            except Exception:
                client = None

    # Helper to create an async wrapper around various client call signatures
    import asyncio

    def _normalize_response(res):
        if isinstance(res, str):
            return _CompatResponse(res)
        text = getattr(res, "text", None) or getattr(res, "output", None) or getattr(res, "content", None)
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        if text is None and hasattr(res, "outputs"):
            outs = getattr(res, "outputs")
            if isinstance(outs, (list, tuple)) and len(outs) > 0:
                out0 = outs[0]
                text = getattr(out0, "text", None) or getattr(out0, "content", None)
                if text is None and isinstance(out0, dict):
                    text = out0.get("text") or out0.get("content")
        if text is None:
            text = str(res)
        return _CompatResponse(text)

    def _normalize_config(config):
        if config is None:
            return None
        if hasattr(config, "dict") and callable(getattr(config, "dict")):
            try:
                return config.dict(exclude_none=True)
            except TypeError:
                return config.dict()
        if hasattr(config, "model_dump") and callable(getattr(config, "model_dump")):
            try:
                return config.model_dump(exclude_none=True)
            except TypeError:
                return config.model_dump()
        return config

    normalized_config = _normalize_config(generation_config)

    def _get_api_key():
        return os.environ.get("GOOGLE_API_KEY")

    def _instantiate_client():
        ClientCls = getattr(GENAI, "Client", None)
        if not ClientCls:
            return getattr(GENAI, "_client", None)
        try:
            client = ClientCls(api_key=_get_api_key())
            setattr(GENAI, "_client", client)
            return client
        except Exception:
            return getattr(GENAI, "_client", None)

    def _get_client():
        client = getattr(GENAI, "_client", None)
        if client is not None:
            return client
        return _instantiate_client()

    def _reset_client():
        if hasattr(GENAI, "_client"):
            try:
                delattr(GENAI, "_client")
            except Exception:
                pass
        return _instantiate_client()

    def _is_transient_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return (
            "503" in message
            or "unavailable" in message
            or "high demand" in message
            or "client has been closed" in message
        )

    client = _get_client()
    models_api = getattr(client, "models", None) if client is not None else None
    if models_api:
        for method_name in ("generate_content", "generate_content_stream", "generate", "generate_text"):
            async def _gen_async(prompt: str, _method_name=method_name):
                import asyncio
                attempt = 0
                last_exc = None
                while attempt < 3:
                    attempt += 1

                    def _call():
                        current_client = _get_client()
                        current_models = getattr(current_client, "models", None)
                        if not current_models:
                            raise RuntimeError("No compatible models API available on the client")
                        current_method = getattr(current_models, _method_name, None)
                        if not callable(current_method):
                            raise RuntimeError(f"No compatible method '{_method_name}' on models API")

                        kwargs = {"model": model_name, "contents": prompt}
                        if normalized_config is not None:
                            kwargs["config"] = normalized_config
                        try:
                            return current_method(**kwargs)
                        except TypeError:
                            try:
                                return current_method(model=model_name, input=prompt, config=generation_config)
                            except TypeError:
                                try:
                                    return current_method(model=model_name, prompt=prompt, config=generation_config)
                                except TypeError:
                                    try:
                                        return current_method(model_name, prompt)
                                    except TypeError:
                                        return current_method(prompt)

                    try:
                        res = await asyncio.to_thread(_call)
                        return _normalize_response(res)
                    except RuntimeError as exc:
                        if "client has been closed" in str(exc).lower():
                            _reset_client()
                            last_exc = exc
                            await asyncio.sleep(1)
                            continue
                        raise
                    except Exception as exc:
                        if attempt < 3 and _is_transient_error(exc):
                            last_exc = exc
                            await asyncio.sleep(2 ** (attempt - 1))
                            continue
                        raise

                raise last_exc or RuntimeError("Failed to execute model call after retries")

            class _ModelWrapper:
                async def generate_content_async(self, prompt: str):
                    return await _gen_async(prompt)

            return _ModelWrapper()

        # Try other client surfaces for backwards compatibility
        for method_name in ("generate", "generate_text", "responses", "chat", "create_text"):
            method = getattr(client, method_name, None)
            if callable(method):
                async def _gen_async(prompt: str, _method=method):
                    # Run sync client calls in a thread
                    def _call():
                        try:
                            # Some clients accept a dict or prompt string
                            return _method(prompt)
                        except TypeError:
                            # Try a different signature
                            return _method(model=model_name, prompt=prompt)
                    res = await asyncio.to_thread(_call)
                    return _normalize_response(res)

                class _ModelWrapper:
                    async def generate_content_async(self, prompt: str):
                        return await _gen_async(prompt)

                return _ModelWrapper()

    # Support newer API surface: GENAI.models.generate(...) or GENAI.client.models.generate(...)
    models_api = getattr(GENAI, "models", None)
    if models_api:
        for method_name in ("generate", "predict", "generate_text"):
            method = getattr(models_api, method_name, None)
            if callable(method):
                async def _gen_async3(prompt: str, _method=method):
                    import asyncio
                    def _call():
                        # Try multiple possible signatures used across versions
                        try:
                            return _method(model=model_name, input=prompt)
                        except TypeError:
                            try:
                                return _method(model=model_name, prompt=prompt)
                            except TypeError:
                                try:
                                    return _method(model_name, prompt)
                                except TypeError:
                                    return _method(prompt)
                    res = await asyncio.to_thread(_call)
                    # Normalize to object with .text
                    if isinstance(res, str):
                        return _CompatResponse(res)
                    text = None
                    # common v1 response shapes
                    if hasattr(res, "outputs"):
                        outs = getattr(res, "outputs")
                        if isinstance(outs, (list, tuple)) and len(outs) > 0:
                            out0 = outs[0]
                            text = getattr(out0, "text", None) or getattr(out0, "content", None)
                            if text is None and isinstance(out0, dict):
                                text = out0.get("text") or out0.get("content")
                    if text is None:
                        text = getattr(res, "output", None) or getattr(res, "response", None) or getattr(res, "text", None)
                    if isinstance(text, (list, tuple)):
                        text = text[0] if text else ""
                    if text is None:
                        text = str(res)
                    return _CompatResponse(text)

                class _ModelWrapper3:
                    async def generate_content_async(self, prompt: str):
                        return await _gen_async3(prompt)

                return _ModelWrapper3()

    # If module exposes a client attribute with models subresource (lowercase 'client')
    client_obj = getattr(GENAI, "client", None)
    if client_obj and getattr(client_obj, "models", None):
        models_api = getattr(client_obj, "models")
        for method_name in ("generate", "predict", "generate_text"):
            method = getattr(models_api, method_name, None)
            if callable(method):
                async def _gen_async4(prompt: str, _method=method):
                    import asyncio
                    def _call():
                        try:
                            return _method(model=model_name, input=prompt)
                        except TypeError:
                            try:
                                return _method(model=model_name, prompt=prompt)
                            except TypeError:
                                try:
                                    return _method(model_name, prompt)
                                except TypeError:
                                    return _method(prompt)
                    res = await asyncio.to_thread(_call)
                    if isinstance(res, str):
                        return _CompatResponse(res)
                    text = None
                    if hasattr(res, "outputs"):
                        outs = getattr(res, "outputs")
                        if isinstance(outs, (list, tuple)) and len(outs) > 0:
                            out0 = outs[0]
                            text = getattr(out0, "text", None) or getattr(out0, "content", None)
                            if text is None and isinstance(out0, dict):
                                text = out0.get("text") or out0.get("content")
                    if text is None:
                        text = getattr(res, "output", None) or getattr(res, "response", None) or getattr(res, "text", None)
                    if isinstance(text, (list, tuple)):
                        text = text[0] if text else ""
                    if text is None:
                        text = str(res)
                    return _CompatResponse(text)

                class _ModelWrapper4:
                    async def generate_content_async(self, prompt: str):
                        return await _gen_async4(prompt)

                return _ModelWrapper4()

    # Last resort: if GENAI exposes a top-level generate function
    for fn_name in ("generate", "generate_text", "responses", "chat"):
        fn = getattr(GENAI, fn_name, None)
        if callable(fn):
            async def _gen_async2(prompt: str, _fn=fn):
                import asyncio
                def _call():
                    try:
                        return _fn(prompt)
                    except TypeError:
                        return _fn(model=model_name, prompt=prompt)
                res = await asyncio.to_thread(_call)
                if isinstance(res, str):
                    return _CompatResponse(res)
                text = getattr(res, "text", None) or getattr(res, "output", None) or getattr(res, "content", None)
                if isinstance(text, (list, tuple)):
                    text = text[0] if text else ""
                if text is None:
                    text = str(res)
                return _CompatResponse(text)

            class _ModelWrapper2:
                async def generate_content_async(self, prompt: str):
                    return await _gen_async2(prompt)

            return _ModelWrapper2()

    # Build a helpful diagnostic message listing top-level attributes
    available_attrs = sorted([a for a in dir(GENAI) if not a.startswith("_")])
    diagnostic = (
        "Unable to construct a compatible model object from the installed "
        "Google generative AI library. Detected public attributes on the module: "
        f"{available_attrs[:50]}"
    )
    diagnostic += (
        "\nIf you are using a newer API surface, consider updating the compatibility "
        "shim in app/services/genai_compat.py to adapt to the library. Ensure the "
        "environment variable GOOGLE_API_KEY is set or call configure_genai(api_key)."
    )
    raise RuntimeError(diagnostic)
