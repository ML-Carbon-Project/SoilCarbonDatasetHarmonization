from __future__ import annotations

import csv
import hashlib
import math
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import Point, shape
from shapely.prepared import prep
from shapely.strtree import STRtree

from carbono_solo.collector.io import fetch_bytes, write_csv


IBGE_BIOMES_EDITION = "2025"
IBGE_BIOMES_SCALE = "1:250000"
IBGE_BIOMES_ARCHIVE_NAME = (
    "2025_Biomas-e-Sistema-Costeiro-Marinho-do-Brasil-1-250000_shp.zip"
)
IBGE_BIOMES_SOURCE_URL = (
    "https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/"
    f"biomas/vetores/{IBGE_BIOMES_ARCHIVE_NAME}"
)
IBGE_BIOMES_ARCHIVE_SHA256 = (
    "247B15C427070805044EBE02012F15552850D535FBFECADDB359875CDDA1EA8F"
)
IBGE_BIOMES_SHAPEFILE_NAME = "lml_bioma_e250k_v20250911_A.shp"
IBGE_BIOMES_REQUIRED_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")

IBGE_TERRITORY_EDITION = "2025"
IBGE_TERRITORY_ARCHIVE_NAME = "BR_Pais_2025.zip"
IBGE_TERRITORY_SOURCE_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
    f"malhas_municipais/municipio_2025/Brasil/{IBGE_TERRITORY_ARCHIVE_NAME}"
)
IBGE_TERRITORY_ARCHIVE_SHA256 = (
    "2DBA5243C9AAC0C6707B8E2E15691DE61E5DE0E065472F17420C5594F1D896C6"
)
IBGE_TERRITORY_SHAPEFILE_NAME = "BR_Pais_2025.shp"
IBGE_TERRITORY_REQUIRED_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")

AUDIT_FIELDS = [
    "LATITUDE",
    "LONGITUDE",
    "BIOME",
    "OUTPUT_STATUS",
    "MATCH_METHOD",
    "CANDIDATE_COUNT",
    "ROW_COUNT",
    "IBGE_EDITION",
    "IBGE_SCALE",
    "LAYER_CRS",
    "SOURCE_URL",
    "ARCHIVE_SHA256",
]


@dataclass(frozen=True)
class BiomeMatch:
    biome: str
    method: str
    candidate_count: int


class IBGEBiomeClassifier:
    def __init__(
        self,
        geometries: list[object],
        biome_names: list[str],
        transformer: Transformer,
        layer_crs: CRS,
    ) -> None:
        if not geometries or len(geometries) != len(biome_names):
            raise ValueError("Biome geometries and names must be non-empty and aligned")
        self._geometries = geometries
        self._prepared_geometries = [prep(geometry) for geometry in geometries]
        self._biome_names = biome_names
        self._transformer = transformer
        self._tree = STRtree(geometries)
        self.layer_crs = layer_crs

    @classmethod
    def from_shapefile(cls, shapefile_path: str | Path) -> "IBGEBiomeClassifier":
        path = Path(shapefile_path)
        if not path.exists():
            raise FileNotFoundError(f"Biome shapefile not found: {path}")

        layer_crs = _read_layer_crs(path)
        transformer = Transformer.from_crs("EPSG:4326", layer_crs, always_xy=True)
        encoding = _read_shapefile_encoding(path)
        reader = shapefile.Reader(str(path), encoding=encoding)
        field_names = [field[0] for field in reader.fields[1:]]
        biome_field = _biome_field_name(field_names)

        geometries = []
        biome_names = []
        for shape_record in reader.iterShapeRecords():
            values = dict(zip(field_names, shape_record.record))
            biome = str(values.get(biome_field, "")).strip()
            if not biome:
                continue
            geometry = shape(shape_record.shape.__geo_interface__)
            if geometry.is_empty:
                continue
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if geometry.is_empty:
                continue
            geometries.append(geometry)
            biome_names.append(biome)

        return cls(geometries, biome_names, transformer, layer_crs)

    def classify(self, latitude: float, longitude: float) -> BiomeMatch:
        point = self._point(latitude, longitude)
        candidate_indices = [int(index) for index in self._tree.query(point)]
        covered_indices = [
            index
            for index in candidate_indices
            if self._prepared_geometries[index].covers(point)
        ]
        if not covered_indices:
            return BiomeMatch("", "unmatched", 0)

        interior_indices = [
            index
            for index in covered_indices
            if self._prepared_geometries[index].contains(point)
        ]
        if len(interior_indices) == 1:
            index = interior_indices[0]
            return BiomeMatch(
                self._biome_names[index],
                "point_in_polygon",
                len(covered_indices),
            )

        eligible_indices = interior_indices or covered_indices
        if len(eligible_indices) == 1:
            index = eligible_indices[0]
            return BiomeMatch(
                self._biome_names[index],
                "point_on_boundary",
                len(covered_indices),
            )

        index = self._tiebreak(point, eligible_indices)
        method = "overlap_tiebreak" if interior_indices else "boundary_tiebreak"
        return BiomeMatch(
            self._biome_names[index],
            method,
            len(covered_indices),
        )

    def nearest(self, latitude: float, longitude: float) -> BiomeMatch:
        point = self._point(latitude, longitude)
        index = int(self._tree.nearest(point))
        return BiomeMatch(
            self._biome_names[index],
            "territory_nearest_biome",
            1,
        )

    def _point(self, latitude: float, longitude: float) -> Point:
        x, y = self._transformer.transform(longitude, latitude)
        return Point(x, y)

    def _tiebreak(self, point: Point, indices: list[int]) -> int:
        epsilon = 1e-5 if self.layer_crs.is_geographic else 1.0
        neighborhood = point.buffer(epsilon)
        ranked = [
            (
                self._geometries[index].intersection(neighborhood).area,
                self._biome_names[index],
                index,
            )
            for index in indices
        ]
        return max(ranked, key=lambda item: (item[0], item[1]))[2]


class BrazilTerritoryMask:
    def __init__(
        self,
        geometries: list[object],
        transformer: Transformer,
        layer_crs: CRS,
    ) -> None:
        if not geometries:
            raise ValueError("Brazil territory layer must contain at least one geometry")
        self._geometries = geometries
        self._prepared_geometries = [prep(geometry) for geometry in geometries]
        self._transformer = transformer
        self._tree = STRtree(geometries)
        self.layer_crs = layer_crs

    @classmethod
    def from_shapefile(cls, shapefile_path: str | Path) -> "BrazilTerritoryMask":
        path = Path(shapefile_path)
        if not path.exists():
            raise FileNotFoundError(f"Brazil territory shapefile not found: {path}")

        layer_crs = _read_layer_crs(path)
        transformer = Transformer.from_crs("EPSG:4326", layer_crs, always_xy=True)
        encoding = _read_shapefile_encoding(path)
        reader = shapefile.Reader(str(path), encoding=encoding)
        geometries = []
        for item in reader.iterShapes():
            geometry = shape(item.__geo_interface__)
            if geometry.is_empty:
                continue
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if not geometry.is_empty:
                geometries.append(geometry)
        return cls(geometries, transformer, layer_crs)

    def covers(self, latitude: float, longitude: float) -> bool:
        x, y = self._transformer.transform(longitude, latitude)
        point = Point(x, y)
        return any(
            self._prepared_geometries[int(index)].covers(point)
            for index in self._tree.query(point)
        )


def ensure_ibge_biome_layer(
    layer_dir: str | Path,
    *,
    allow_download: bool = True,
    source_url: str = IBGE_BIOMES_SOURCE_URL,
    expected_sha256: str = IBGE_BIOMES_ARCHIVE_SHA256,
) -> Path:
    return _ensure_pinned_shapefile(
        layer_dir=layer_dir,
        archive_name=IBGE_BIOMES_ARCHIVE_NAME,
        shapefile_name=IBGE_BIOMES_SHAPEFILE_NAME,
        required_suffixes=IBGE_BIOMES_REQUIRED_SUFFIXES,
        allow_download=allow_download,
        source_url=source_url,
        expected_sha256=expected_sha256,
    )


def ensure_ibge_territory_layer(
    layer_dir: str | Path,
    *,
    allow_download: bool = True,
    source_url: str = IBGE_TERRITORY_SOURCE_URL,
    expected_sha256: str = IBGE_TERRITORY_ARCHIVE_SHA256,
) -> Path:
    return _ensure_pinned_shapefile(
        layer_dir=layer_dir,
        archive_name=IBGE_TERRITORY_ARCHIVE_NAME,
        shapefile_name=IBGE_TERRITORY_SHAPEFILE_NAME,
        required_suffixes=IBGE_TERRITORY_REQUIRED_SUFFIXES,
        allow_download=allow_download,
        source_url=source_url,
        expected_sha256=expected_sha256,
    )


def enrich_biome_csv(
    input_path: str | Path,
    output_path: str | Path,
    audit_path: str | Path,
    shapefile_path: str | Path,
    territory_shapefile_path: str | Path,
) -> dict[str, int]:
    fields, rows = _read_csv(input_path)
    _validate_input_fields(fields)
    classifier = IBGEBiomeClassifier.from_shapefile(shapefile_path)
    territory = BrazilTerritoryMask.from_shapefile(territory_shapefile_path)

    point_rows: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key, latitude, longitude = _coordinate_key(row)
        point = point_rows.setdefault(
            key,
            {
                "latitude_text": _clean(row.get("LATITUDE")),
                "longitude_text": _clean(row.get("LONGITUDE")),
                "latitude": latitude,
                "longitude": longitude,
                "row_count": 0,
            },
        )
        point["row_count"] = int(point["row_count"]) + 1

    matches: dict[tuple[object, ...], BiomeMatch] = {}
    audit_rows = []
    for key, point in point_rows.items():
        latitude = point["latitude"]
        longitude = point["longitude"]
        if latitude is None or longitude is None:
            match = BiomeMatch("", "removed_invalid_coordinates", 0)
        elif not territory.covers(float(latitude), float(longitude)):
            match = BiomeMatch("", "outside_brazil_or_at_sea", 0)
        else:
            match = classifier.classify(float(latitude), float(longitude))
            if not match.biome:
                match = classifier.nearest(float(latitude), float(longitude))
        matches[key] = match
        audit_rows.append(
            {
                "LATITUDE": point["latitude_text"],
                "LONGITUDE": point["longitude_text"],
                "BIOME": match.biome,
                "OUTPUT_STATUS": (
                    "kept"
                    if match.biome
                    else (
                        "removed_invalid_coordinates"
                        if match.method == "removed_invalid_coordinates"
                        else "removed_outside_brazil_or_at_sea"
                    )
                ),
                "MATCH_METHOD": match.method,
                "CANDIDATE_COUNT": str(match.candidate_count),
                "ROW_COUNT": str(point["row_count"]),
                "IBGE_EDITION": IBGE_BIOMES_EDITION,
                "IBGE_SCALE": IBGE_BIOMES_SCALE,
                "LAYER_CRS": classifier.layer_crs.to_string(),
                "SOURCE_URL": IBGE_BIOMES_SOURCE_URL,
                "ARCHIVE_SHA256": IBGE_BIOMES_ARCHIVE_SHA256,
            }
        )

    enriched_rows = []
    for row in rows:
        key, _latitude, _longitude = _coordinate_key(row)
        if not matches[key].biome:
            continue
        enriched = dict(row)
        enriched["BIOME"] = matches[key].biome
        enriched_rows.append(enriched)

    output_fields = _fields_with_biome(fields)
    _atomic_write_csv(output_path, output_fields, enriched_rows)
    _atomic_write_csv(audit_path, AUDIT_FIELDS, audit_rows)

    matched_points = sum(1 for match in matches.values() if match.biome)
    removed_rows = len(rows) - len(enriched_rows)
    return {
        "input_rows": len(rows),
        "output_rows": len(enriched_rows),
        "unique_points": len(matches),
        "matched_points": matched_points,
        "removed_points": len(matches) - matched_points,
        "removed_rows": removed_rows,
        "territory_fallback_points": sum(
            1
            for match in matches.values()
            if match.method == "territory_nearest_biome"
        ),
        "boundary_points": sum(
            1 for match in matches.values() if "boundary" in match.method
        ),
        "matched_rows": sum(
            int(point_rows[key]["row_count"])
            for key, match in matches.items()
            if match.biome
        ),
    }


def _read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def _validate_input_fields(fields: list[str]) -> None:
    missing = [field for field in ("LATITUDE", "LONGITUDE", "STATE") if field not in fields]
    if missing:
        raise ValueError(f"Harmonized CSV is missing required fields: {', '.join(missing)}")


def _fields_with_biome(fields: list[str]) -> list[str]:
    if "BIOME" in fields:
        return fields.copy()
    output_fields = fields.copy()
    state_index = output_fields.index("STATE")
    output_fields.insert(state_index + 1, "BIOME")
    return output_fields


def _coordinate_key(
    row: Mapping[str, object],
) -> tuple[tuple[object, ...], float | None, float | None]:
    latitude_text = _clean(row.get("LATITUDE"))
    longitude_text = _clean(row.get("LONGITUDE"))
    latitude = _to_float(latitude_text)
    longitude = _to_float(longitude_text)
    if latitude is None or longitude is None:
        return ("invalid", latitude_text, longitude_text), latitude, longitude
    return (latitude, longitude), latitude, longitude


def _to_float(value: object) -> float | None:
    text = _clean(value).replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _read_layer_crs(shapefile_path: Path) -> CRS:
    prj_path = shapefile_path.with_suffix(".prj")
    if not prj_path.exists():
        raise FileNotFoundError(f"Biome layer CRS file not found: {prj_path}")
    return CRS.from_wkt(prj_path.read_text(encoding="utf-8-sig"))


def _read_shapefile_encoding(shapefile_path: Path) -> str:
    cpg_path = shapefile_path.with_suffix(".cpg")
    if not cpg_path.exists():
        return "utf-8"
    encoding = cpg_path.read_text(encoding="ascii", errors="ignore").strip()
    return encoding or "utf-8"


def _biome_field_name(field_names: list[str]) -> str:
    normalized_fields = {_normalize_name(name): name for name in field_names}
    for candidate in ("BIOMA", "NOM_BIOMA", "NM_BIOMA", "NOME_BIOMA"):
        if candidate in normalized_fields:
            return normalized_fields[candidate]
    raise ValueError(
        "Could not identify the biome-name field in IBGE layer. "
        f"Available fields: {', '.join(field_names)}"
    )


def _normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if not unicodedata.combining(character)).upper()


def _ensure_pinned_shapefile(
    *,
    layer_dir: str | Path,
    archive_name: str,
    shapefile_name: str,
    required_suffixes: tuple[str, ...],
    allow_download: bool,
    source_url: str,
    expected_sha256: str,
) -> Path:
    directory = Path(layer_dir)
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / archive_name
    shapefile_path = directory / shapefile_name

    if not archive_path.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"IBGE archive not found in offline mode: {archive_path}"
            )
        payload = fetch_bytes(source_url, timeout=180)
        actual_sha256 = hashlib.sha256(payload).hexdigest().upper()
        _validate_sha256(actual_sha256, expected_sha256, source_url)
        temporary_archive = archive_path.with_suffix(archive_path.suffix + ".part")
        temporary_archive.write_bytes(payload)
        temporary_archive.replace(archive_path)

    actual_sha256 = _sha256_file(archive_path)
    _validate_sha256(actual_sha256, expected_sha256, str(archive_path))

    if not _required_layer_files_exist_for(shapefile_path, required_suffixes):
        _extract_zip_safely(archive_path, directory)
    if not _required_layer_files_exist_for(shapefile_path, required_suffixes):
        raise RuntimeError(
            f"IBGE archive does not contain the required shapefile: {shapefile_path}"
        )
    return shapefile_path


def _required_layer_files_exist_for(
    shapefile_path: Path,
    required_suffixes: tuple[str, ...],
) -> bool:
    return all(
        shapefile_path.with_suffix(suffix).exists()
        for suffix in required_suffixes
    )


def _extract_zip_safely(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_resolved != member_path and destination_resolved not in member_path.parents:
                raise RuntimeError(f"Unsafe path in biome archive: {member.filename}")
        archive.extractall(destination)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_sha256(actual: str, expected: str, source: str) -> None:
    if actual.upper() != expected.upper():
        raise RuntimeError(
            f"Unexpected SHA-256 for IBGE biome archive from {source}: "
            f"expected {expected.upper()}, got {actual.upper()}"
        )


def _atomic_write_csv(
    path: str | Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    output_path = Path(path)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    write_csv(temporary_path, fieldnames, rows, encoding="utf-8-sig")
    temporary_path.replace(output_path)
