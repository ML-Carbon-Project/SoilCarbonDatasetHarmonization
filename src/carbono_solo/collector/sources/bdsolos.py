from __future__ import annotations

import csv
import io
import math
import re
import unicodedata
from http.cookiejar import CookieJar
from typing import Protocol
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener

from carbono_solo.collector.geo import is_inside_brazil_bbox, normalize_year
from carbono_solo.collector.io import USER_AGENT
from carbono_solo.collector.models import (
    CATALOG_ONLY,
    FAILED,
    IMPLEMENTED,
    InventoryRecord,
    SourceResult,
    TargetRecord,
)


SOURCE_ID = "bdsolos_embrapa"
SOURCE_NAME = "SISolos / BD Solos Embrapa"
DATASET_ID = "bdsolos_public_carbono_organico"
BASE_URL = "https://www.bdsolos.cnptia.embrapa.br"
QUERY_URL = f"{BASE_URL}/consultas/sistema/consultas.gerador.query.php"
EXPORT_URL = f"{BASE_URL}/consultas/sistema/consultas.gerador.query.resultados.php"
MAIN_URL = f"{BASE_URL}/main.php"


class BdSolosClient(Protocol):
    def initialize(self) -> None:
        ...

    def post_text(self, url: str, body: str, timeout: int = 60) -> str:
        ...

    def fetch_text(self, url: str, timeout: int = 60) -> str:
        ...


def collect_bdsolos_embrapa(
    *,
    retrieved_at: str,
    use_network: bool = True,
    batch_size: int = 20,
    client: BdSolosClient | None = None,
) -> SourceResult:
    if not use_network:
        return SourceResult(inventory=[], targets=[])

    try:
        client = client or StatefulBdSolosClient()
        client.initialize()
        query_html = client.post_text(QUERY_URL, _query_body(), timeout=180)
        result_ids = _result_ids(query_html)
        targets: list[TargetRecord] = []
        for batch in _batches(result_ids, batch_size):
            token = client.post_text(EXPORT_URL, _export_body(batch), timeout=180).strip()
            if not token:
                continue
            if not re.fullmatch(r"[a-f0-9]{40,}", token):
                raise ValueError(f"Unexpected BD Solos export token: {token}")
            csv_text = client.fetch_text(_csv_url(token), timeout=180)
            targets.extend(targets_from_csv(csv_text, source_file=f"{token}.csv"))

        return SourceResult(
            inventory=[
                _inventory_record(
                    retrieved_at=retrieved_at,
                    status=IMPLEMENTED if targets else CATALOG_ONLY,
                    notes=(
                        f"Parsed {len(targets)} target rows from public BD Solos "
                        f"CSV exports across {len(result_ids)} selected trabalhos."
                    ),
                )
            ],
            targets=targets,
        )
    except Exception as exc:
        return SourceResult(
            inventory=[
                _inventory_record(
                    retrieved_at=retrieved_at,
                    status=FAILED,
                    notes=f"Ingestion error: {exc}",
                )
            ],
            targets=[],
        )


class StatefulBdSolosClient:
    def __init__(self) -> None:
        self._opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def initialize(self) -> None:
        self.post_text(
            MAIN_URL,
            urlencode({"ini": "0", "consulta_publica": "1"}),
            timeout=60,
        )

    def post_text(self, url: str, body: str, timeout: int = 60) -> str:
        request = Request(
            url,
            data=body.encode("utf-8"),
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            },
        )
        return self._open_text(request, timeout=timeout)

    def fetch_text(self, url: str, timeout: int = 60) -> str:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        return self._open_text(request, timeout=timeout)

    def _open_text(self, request: Request, timeout: int) -> str:
        try:
            with self._opener.open(request, timeout=timeout) as response:
                headers = getattr(response, "headers", None)
                get_content_charset = getattr(headers, "get_content_charset", None)
                charset = get_content_charset() if get_content_charset else None
                payload = response.read()
        except HTTPError as exc:
            reason = getattr(exc, "reason", None) or getattr(exc, "msg", None)
            detail = f"{exc.code} {reason}" if reason else str(exc.code)
            raise RuntimeError(f"HTTP error fetching {request.full_url}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Error fetching {request.full_url}: {exc}") from exc

        try:
            return payload.decode(charset or "utf-8-sig")
        except UnicodeDecodeError:
            return payload.decode("latin-1")


def targets_from_csv(text: str, *, source_file: str) -> list[TargetRecord]:
    reader = csv.DictReader(
        io.StringIO(_csv_payload(text)),
        delimiter=";",
        quotechar='"',
    )
    records: list[TargetRecord] = []
    for raw_row in reader:
        row = {_normalize_key(key): _clean_text(value) for key, value in raw_row.items()}
        latitude, longitude = _coordinates(row)
        if latitude is None or longitude is None:
            continue
        if not is_inside_brazil_bbox(latitude=latitude, longitude=longitude):
            continue

        depth_top = _to_float(_value(row, "Profundidade Superior"))
        depth_bottom = _to_float(_value(row, "Profundidade Inferior"))
        carbon = _to_float(_value(row, "Carbono orgânico"))
        if depth_top is None or depth_bottom is None or carbon is None:
            continue

        code_work = _value(row, "Código Trabalho")
        point_id = _value(row, "Código PA")
        horizon_id = _value(row, "Código Horizonte")
        observation_id = ":".join(
            part for part in (code_work, point_id, horizon_id) if part
        )
        if not observation_id:
            continue

        sample_date, sample_year = _sample_period(_value(row, "Data da Coleta"))
        depth_top_text = _format_number(depth_top)
        depth_bottom_text = _format_number(depth_bottom)
        carbon_text = _format_number(carbon)
        density = _to_float(_value(row, "Densidade - Solo (aparente)"))
        coarse = _coarse_fraction(row)
        unified_id = (
            f"{SOURCE_ID}:{DATASET_ID}:{source_file}:{observation_id}:"
            f"{depth_top_text}-{depth_bottom_text}:carbon_content"
        )

        records.append(
            TargetRecord(
                unified_observation_id=unified_id,
                source_id=SOURCE_ID,
                dataset_id=DATASET_ID,
                source_observation_id=observation_id,
                latitude=latitude,
                longitude=longitude,
                coord_x=longitude,
                coord_y=latitude,
                crs="EPSG:4326",
                country="Brazil",
                state=_value(row, "UF"),
                sample_date=sample_date,
                sample_year=sample_year,
                sample_period_start=sample_year,
                sample_period_end=sample_year,
                depth_top_cm=depth_top_text,
                depth_bottom_cm=depth_bottom_text,
                layer_thickness_cm=_format_number(depth_bottom - depth_top),
                carbon_content_g_kg=carbon_text,
                carbon_stock_kg_m2="",
                carbon_stock_mg_ha="",
                bulk_density_g_cm3="" if density is None else _format_number(density),
                coarse_fragments_fraction=(
                    "" if coarse is None else _format_number(coarse)
                ),
                carbon_stock_00_05="",
                carbon_stock_05_15="",
                carbon_stock_15_30="",
                carbon_stock_30_60="",
                carbon_stock_60_100="",
                target_kind="carbon_content",
                target_value=carbon_text,
                target_unit="g/kg",
                derivation_method="reported_bdsolos_carbono_organico_g_kg",
                quality_flag="ok",
                duplicate_group_id="",
                duplicate_status="active",
                duplicate_resolution="unique",
            )
        )
    return records


def _query_body() -> str:
    return urlencode(
        {
            "tr": "0",
            "q": "4",
            "set": "7",
            "set_tr": "1",
            "set_pa": "3",
            "set_hz": "13",
            "set_um": "0",
            "set_cp": "0",
            "hz_pq[i][hz_qco]": "0",
            "hz_pq[o][hz_qco]": "4",
            "flag_tr": "0",
            "flag_pa": "0",
            "flag_hz": "4",
            "flag_c": "0",
            "flag_um": "0",
        }
    )


def _export_body(result_ids: list[str]) -> str:
    return urlencode(
        {
            "t": "csv",
            "res": f" {'|'.join(result_ids)}",
            "sel": "7",
            "s_tr": "1",
            "s_pa": "3",
            "s_hz": "13",
            "s_um": "0",
            "s_cp": "0",
        }
    )


def _csv_url(token: str) -> str:
    return f"{BASE_URL}/csv/{token}/{token}.csv"


def _result_ids(html: str) -> list[str]:
    ids = re.findall(r'id=["\']info\[\d+\]["\'][^>]*value=["\'](\d+)["\']', html)
    return list(dict.fromkeys(ids))


def _csv_payload(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "codigo trabalho" in _normalize_key(line):
            return "\n".join(lines[index:])
    return text


def _coordinates(row: dict[str, str]) -> tuple[float | None, float | None]:
    latitude = _dms_value(
        degrees=_value(row, "Latitude Graus"),
        minutes=_value(row, "Latitude Minutos"),
        seconds=_value(row, "Latitude Segundos"),
        hemisphere=_value(row, "Latitude Hemisfério"),
        default_negative=True,
    )
    longitude = _dms_value(
        degrees=_value(row, "Longitude Graus"),
        minutes=_value(row, "Longitude Minutos"),
        seconds=_value(row, "Longitude Segundos"),
        hemisphere=_value(row, "Longitude Hemisfério"),
        default_negative=True,
    )
    return latitude, longitude


def _dms_value(
    *,
    degrees: str,
    minutes: str,
    seconds: str,
    hemisphere: str,
    default_negative: bool,
) -> float | None:
    degree_value = _to_float(degrees)
    if degree_value is None:
        return None
    minute_value = _to_float(minutes) or 0.0
    second_value = _to_float(seconds) or 0.0
    coordinate = abs(degree_value) + minute_value / 60.0 + second_value / 3600.0
    normalized_hemisphere = _normalize_key(hemisphere)
    if normalized_hemisphere in {"sul", "oeste", "s", "w", "o"}:
        coordinate *= -1
    elif degree_value < 0 or (not normalized_hemisphere and default_negative):
        coordinate *= -1
    return coordinate


def _coarse_fraction(row: dict[str, str]) -> float | None:
    calhaus = _to_float(_value(row, "Frações da Amostra Total - Calhaus (g/Kg)")) or 0.0
    cascalho = _to_float(_value(row, "Frações da Amostra Total - Cascalho (g/Kg)")) or 0.0
    total = calhaus + cascalho
    if total <= 0:
        return None
    return total / 1000.0


def _sample_period(raw_date: str) -> tuple[str, str]:
    text = _clean_text(raw_date)
    date_match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", text)
    if date_match:
        day, month, year = (int(part) for part in date_match.groups())
        return f"{year:04d}-{month:02d}-{day:02d}", normalize_year(year)
    year = normalize_year(text)
    return "", year


def _inventory_record(*, retrieved_at: str, status: str, notes: str) -> InventoryRecord:
    return InventoryRecord(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        dataset_id=DATASET_ID,
        title="BD Solos public query filtered by organic carbon",
        url=BASE_URL,
        country_scope="Brazil",
        access_type="public_web_form_csv_export",
        brazil_filter="native_brazil_system",
        candidate_target_fields=(
            "organic carbon, layer depth, coordinates, bulk density, coarse fragments"
        ),
        target_fit_score="high",
        implementation_status=status,
        license="Embrapa public query terms",
        notes=notes,
        retrieved_at=retrieved_at,
    )


def _batches(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _value(row: dict[str, str], key: str) -> str:
    return row.get(_normalize_key(key), "")


def _to_float(value: object) -> float | None:
    text = _clean_text(value).replace(",", ".")
    if not text:
        return None
    if text.startswith("<") or text.startswith(">"):
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _clean_text(value: object) -> str:
    return str(value or "").strip().strip('"').strip()


def _normalize_key(value: object) -> str:
    text = _clean_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())
