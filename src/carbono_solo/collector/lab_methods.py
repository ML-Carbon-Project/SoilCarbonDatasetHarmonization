from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from carbono_solo.collector.io import write_csv


NOT_REPORTED = "N\u00e3o informado"
NOT_APPLICABLE_PSEUDO = "N\u00e3o se aplica - ponto pseudoamostral"

DRY_COMBUSTION_ELEMENTAL = "Combust\u00e3o seca - analisador elementar"
DRY_COMBUSTION_THERMO = (
    "Combust\u00e3o seca - analisador elementar C/N Thermo Flash"
)
WET_WALKLEY_BLACK = "Oxida\u00e7\u00e3o \u00famida - Walkley-Black (1934)"
WET_WALKLEY_BLACK_TITRATION = (
    "Oxida\u00e7\u00e3o \u00famida - Walkley-Black (1934), titula\u00e7\u00e3o"
)
WET_WALKLEY_BLACK_SPECTRO = (
    "Oxida\u00e7\u00e3o \u00famida - Walkley-Black (1934), espectrofotometria"
)
WET_MEBIUS_TITRATION = (
    "Oxida\u00e7\u00e3o \u00famida - Walkley-Black modificado/Mebius (1960), "
    "titula\u00e7\u00e3o"
)
WET_MEBIUS = "Oxida\u00e7\u00e3o \u00famida - Walkley-Black modificado/Mebius (1960)"
WET_YEOMANS_BREMNER = "Oxida\u00e7\u00e3o \u00famida - Yeomans-Bremner (1988)"
WET_YEOMANS_MEBIUS = (
    "Oxida\u00e7\u00e3o \u00famida - Yeomans-Bremner (1988) / Mebius em bloco"
)
WET_IAC_COLORIMETRIC_PROXY = (
    "Oxida\u00e7\u00e3o \u00famida com dicromato - IAC (2001), leitura "
    "colorim\u00e9trica (proxy de mat\u00e9ria org\u00e2nica)"
)
RODELLA_1993 = "M\u00e9todo de Rodella (1993) para carbono/mat\u00e9ria org\u00e2nica"
ALLISON_1966 = "Carbono total - Allison et al. (1966)"
DRY_EUROPA_ANCA_IRMS = (
    "Combust\u00e3o seca - analisador elementar Europa ANCA-SL / IRMS Europa 20-20"
)
DRY_NA1108_IRMS = (
    "Combust\u00e3o seca - analisador elementar NA 1108 / IRMS Finnigan MAT delta S"
)
DRY_VARIO_EL = "Combust\u00e3o seca - analisador elementar Elementar Vario EL"
DRY_LECO_CARBON = "Combust\u00e3o seca - LECO Carbon Analyzer"
DRY_ELEMENTAL_CH = "Combust\u00e3o seca - analisador elementar C,H"
DRY_ELEMENTAL_CF_IRMS = (
    "Combust\u00e3o seca - analisador elementar / IRMS de fluxo cont\u00ednuo (CENA)"
)
DRY_TRUSPEC_950 = (
    "Combust\u00e3o seca a 950 \u00b0C - analisador elementar TruSpec CHN LECO"
)


AUDIT_FIELDS = [
    "source_id",
    "dataset_id",
    "carbon_analysis_method",
    "assignment_basis",
    "row_count",
    "evidence_reference",
    "notes",
]


@dataclass(frozen=True)
class MethodAssignment:
    method: str
    basis: str
    evidence_reference: str
    notes: str = ""


SOILDATA_ASSIGNMENTS = {
    "FBSGZZ": MethodAssignment(
        WET_WALKLEY_BLACK,
        "documentacao_do_estudo:FBSGZZ",
        "doi:10.60502/SoilData/FBSGZZ; doi:10.5935/1984-2295.20160112",
    ),
    "NIIMSF": MethodAssignment(
        WET_IAC_COLORIMETRIC_PROXY,
        "metadado_do_dataset:NIIMSF",
        "doi:10.60502/SoilData/NIIMSF; IAC (2001), cap. 9",
        "A fonte reporta materia organica, nao carbono medido diretamente.",
    ),
    "NSZ9WW": MethodAssignment(
        NOT_REPORTED,
        "documentacao_insuficiente:NSZ9WW",
        "doi:10.60502/SoilData/NSZ9WW",
    ),
    "OXT1PI": MethodAssignment(
        DRY_COMBUSTION_THERMO,
        "metadado_do_dataset:OXT1PI",
        "doi:10.60502/SoilData/OXT1PI; arquivo de metodo 295",
    ),
    "R91CFI": MethodAssignment(
        WET_YEOMANS_BREMNER,
        "metadado_e_publicacao:R91CFI",
        "doi:10.60502/SoilData/R91CFI; Yeomans e Bremner (1988)",
    ),
    "RCYUYZ": MethodAssignment(
        WET_YEOMANS_MEBIUS,
        "metadado_do_dataset:RCYUYZ",
        "doi:10.60502/SoilData/RCYUYZ; arquivo de metodo 11",
    ),
    "YOHSMZ": MethodAssignment(
        DRY_COMBUSTION_ELEMENTAL,
        "metadado_do_dataset:YOHSMZ",
        "doi:10.60502/SoilData/YOHSMZ",
    ),
}


ISRAD_DOIS = {
    "Camargo_1999": "doi:10.1046/j.1365-2486.1999.00259.x",
    "Desjardins_1994": "doi:10.1016/0016-7061(94)90013-2",
    "D\u00fcmig_2008": "doi:10.1016/j.geoderma.2007.06.005",
    "Guillet_1988": "doi:10.1016/0031-0182(88)90111-3",
    "James_2019": "doi:10.1016/j.geoderma.2019.03.019",
    "Jungkunst_2023": "doi:10.1038/s41598-023-30801-x",
    "Kuhnen_2019": "doi:10.5281/zenodo.2645510",
    "L\u00e4hteenoja_2009": "doi:10.1111/j.1365-2486.2009.01920.x",
    "Martinez_2023": "doi:10.1111/ejss.13415",
    "McClaran_2000": "doi:10.2307/3236777",
    "Montes_2023": "doi:10.1016/j.catena.2022.106837",
    "Nagy_2017": "doi:10.1002/2017JG004269",
    "Perez_2006": "doi:10.1890/1051-0761(2006)016[2153:NONADN]2.0.CO;2",
    "Pessenda_1996": "doi:10.1017/S0033822200017574",
    "Pessenda_1997": "doi:10.1017/S0033822200018981",
    "Sanaiotti_2002": "doi:10.1111/j.1744-7429.2002.tb00237.x",
    "Scheibe_2022": "doi:10.5194/bg-20-827-2023",
    "Sierra_2013": "doi:10.5194/bg-10-3455-2013",
    "Telles_2003": "doi:10.1029/2002GB001953",
    "Telles_2011": "doi:10.3334/ORNLDAAC/1025",
    "Trumbore_1993": "doi:10.1029/93GB00468",
    "de_Freitas_2001": "doi:10.1006/qres.2000.2192",
}


ISRAD_ASSIGNMENTS = {
    "D\u00fcmig_2008": DRY_VARIO_EL,
    "Jungkunst_2023": DRY_TRUSPEC_950,
    "L\u00e4hteenoja_2009": DRY_LECO_CARBON,
    "Pessenda_1996": DRY_ELEMENTAL_CH,
    "Scheibe_2022": DRY_NA1108_IRMS,
    "Telles_2003": DRY_ELEMENTAL_CF_IRMS,
}


def method_from_text(value: object, *, fallback: str = "") -> str:
    raw = _clean_text(value)
    normalized = _normalize(raw)
    if not normalized:
        return fallback
    if "thermo flash" in normalized:
        return DRY_COMBUSTION_THERMO
    if any(
        token in normalized
        for token in (
            "combustao seca",
            "dry combustion",
            "elemental analyzer",
            "analisador elementar",
        )
    ):
        return DRY_COMBUSTION_ELEMENTAL
    if "walkley" in normalized and "mebius" in normalized and "titul" in normalized:
        return WET_MEBIUS_TITRATION
    if "walkley" in normalized and "mebius" in normalized:
        return WET_MEBIUS
    if "walkley" in normalized and "espectrofot" in normalized:
        return WET_WALKLEY_BLACK_SPECTRO
    if "walkley" in normalized and "titul" in normalized:
        return WET_WALKLEY_BLACK_TITRATION
    if "walkley" in normalized:
        return WET_WALKLEY_BLACK
    if "yeomans" in normalized and "mebius" in normalized:
        return WET_YEOMANS_MEBIUS
    if "yeomans" in normalized:
        return WET_YEOMANS_BREMNER
    if "rodella" in normalized:
        return RODELLA_1993
    if "allisson" in normalized or "allison" in normalized:
        return ALLISON_1966
    return raw


def soildata_assignment(dataset_id: object, source_column: object = "") -> MethodAssignment:
    explicit_method = method_from_text(source_column)
    if explicit_method:
        return MethodAssignment(
            explicit_method,
            "cabecalho_da_variavel",
            str(dataset_id),
        )
    code = _dataset_code(dataset_id)
    return SOILDATA_ASSIGNMENTS.get(
        code,
        MethodAssignment(NOT_REPORTED, "documentacao_insuficiente", str(dataset_id)),
    )


def mapbiomas_assignment(source_observation_id: object) -> MethodAssignment:
    point_id = _clean_text(source_observation_id)
    normalized = point_id.lower()
    if normalized.startswith(("rock-pseudo-", "sand-pseudo-")):
        return MethodAssignment(
            NOT_APPLICABLE_PSEUDO,
            "identificador_de_pseudoamostra",
            "doi:10.60502/SoilData/IUZOAK",
            "Linha sintetica usada no conjunto de modelagem; nao e ensaio laboratorial.",
        )
    if normalized.startswith("ctb0003-"):
        return MethodAssignment(
            WET_YEOMANS_MEBIUS,
            "linhagem_mapbiomas:ctb0003->RCYUYZ",
            "doi:10.60502/SoilData/RCYUYZ; doi:10.60502/SoilData/IUZOAK",
        )
    return MethodAssignment(
        NOT_REPORTED,
        "metodo_ausente_no_arquivo_c3",
        "doi:10.60502/SoilData/IUZOAK",
    )


def israd_assignment(
    entry_name: object,
    sample_year: object = "",
) -> MethodAssignment:
    entry = _clean_text(entry_name)
    year = _clean_text(sample_year)
    if entry == "Nagy_2017" and year == "2013":
        return MethodAssignment(
            DRY_EUROPA_ANCA_IRMS,
            "documentacao_do_estudo:Nagy_2017;amostras_2013",
            ISRAD_DOIS[entry],
            "O metodo foi comprovado apenas para as amostras coletadas em 2013.",
        )
    documented_method = ISRAD_ASSIGNMENTS.get(entry)
    if documented_method:
        return MethodAssignment(
            documented_method,
            f"documentacao_do_estudo:{entry}",
            ISRAD_DOIS.get(entry, ""),
        )
    notes = "O esquema flat layer nao inclui o metodo analitico."
    if entry == "Nagy_2017":
        notes += " A tecnica localizada so foi comprovada para amostras de 2013."
    return MethodAssignment(
        NOT_REPORTED,
        f"metodo_ausente_no_israd_flat_layer:{entry or 'sem_entry_name'}",
        ISRAD_DOIS.get(
            entry,
            "https://github.com/International-Soil-Radiocarbon-Database/ISRaD",
        ),
        notes,
    )


def hybras_assignment(raw_method: object) -> MethodAssignment:
    raw = _clean_text(raw_method)
    method = method_from_text(raw, fallback=NOT_REPORTED)
    return MethodAssignment(
        method,
        "campo_fonte:OC_OM_Method" if raw else "campo_fonte_vazio:OC_OM_Method",
        "data/raw/hybras/v1_2020/HYBRAS_COMPLETE_V1.xlsx; doi:10.2136/vzj2017.05.0095",
        raw,
    )


def resolve_assignment(row: Mapping[str, object]) -> MethodAssignment:
    existing = _clean_text(row.get("carbon_analysis_method"))
    source_id = _clean_text(row.get("source_id"))
    dataset_id = _clean_text(row.get("dataset_id"))
    observation_id = _clean_text(row.get("source_observation_id"))
    if source_id == "soildata_dataset":
        return soildata_assignment(dataset_id)
    if source_id == "mapbiomas_soc":
        return mapbiomas_assignment(observation_id)
    if source_id == "israd_brazil":
        return israd_assignment(
            observation_id.split(":", 1)[0],
            row.get("sample_year"),
        )
    if source_id == "wosis_brazil":
        return MethodAssignment(
            NOT_REPORTED,
            "metodo_analitico_nao_exposto_pelo_wfs",
            "https://docs.isric.org/globaldata/wosis/faq-wosis.html",
        )
    if source_id == "bdsolos_embrapa":
        return MethodAssignment(
            NOT_REPORTED,
            "metodo_analitico_ausente_na_exportacao_publica",
            "https://www.bdsolos.cnptia.embrapa.br/",
        )
    if source_id == "hybras_sgb":
        return MethodAssignment(
            NOT_REPORTED,
            "requer_campo_fonte:OC_OM_Method",
            "data/raw/hybras/v1_2020/HYBRAS_COMPLETE_V1.xlsx",
        )
    if existing and existing != NOT_REPORTED:
        return MethodAssignment(
            method_from_text(existing, fallback=NOT_REPORTED),
            "campo_existente_na_origem",
            dataset_id,
        )
    return MethodAssignment(NOT_REPORTED, "documentacao_insuficiente", dataset_id)


def annotate_target_csvs(
    active_path: Path,
    duplicate_path: Path,
    audit_path: Path,
    hybras_workbook_path: Path | None = None,
) -> dict[str, int]:
    datasets = [
        (active_path, _read_csv(active_path)),
        (duplicate_path, _read_csv(duplicate_path)),
    ]
    hybras_lookup = _hybras_method_lookup(hybras_workbook_path)
    annotated: list[tuple[Path, list[dict[str, str]], list[MethodAssignment]]] = []

    for path, rows in datasets:
        assignments = []
        for row in rows:
            assignment = _assignment_for_existing_row(row, hybras_lookup)
            row["carbon_analysis_method"] = assignment.method
            assignments.append(assignment)
        annotated.append((path, rows, assignments))

    annotated = _propagate_unambiguous_duplicate_methods(annotated)

    audit_counter: Counter[tuple[str, ...]] = Counter()
    for path, rows, assignments in annotated:
        fields = _fields_with_method(rows)
        write_csv(path, fields, rows, encoding="utf-8-sig")
        for row, assignment in zip(rows, assignments):
            key = (
                _clean_text(row.get("source_id")),
                _clean_text(row.get("dataset_id")),
                _clean_text(row.get("carbon_analysis_method")) or NOT_REPORTED,
                assignment.basis,
                assignment.evidence_reference,
                assignment.notes,
            )
            audit_counter[key] += 1

    audit_rows = [
        {
            "source_id": key[0],
            "dataset_id": key[1],
            "carbon_analysis_method": key[2],
            "assignment_basis": key[3],
            "row_count": count,
            "evidence_reference": key[4],
            "notes": key[5],
        }
        for key, count in sorted(audit_counter.items())
    ]
    write_csv(audit_path, AUDIT_FIELDS, audit_rows, encoding="utf-8-sig")

    all_rows = [row for _path, rows, _assignments in annotated for row in rows]
    return {
        "active_rows": len(annotated[0][1]),
        "duplicate_rows": len(annotated[1][1]),
        "not_reported_rows": sum(
            row["carbon_analysis_method"] == NOT_REPORTED for row in all_rows
        ),
        "not_applicable_rows": sum(
            row["carbon_analysis_method"] == NOT_APPLICABLE_PSEUDO for row in all_rows
        ),
        "documented_rows": sum(
            row["carbon_analysis_method"] not in {NOT_REPORTED, NOT_APPLICABLE_PSEUDO}
            for row in all_rows
        ),
        "audit_rows": len(audit_rows),
    }


def _assignment_for_existing_row(
    row: Mapping[str, object],
    hybras_lookup: Mapping[tuple[str, str, str], MethodAssignment],
) -> MethodAssignment:
    if _clean_text(row.get("source_id")) == "hybras_sgb":
        key = (
            _clean_text(row.get("source_observation_id")),
            _number_key(row.get("depth_top_cm")),
            _number_key(row.get("depth_bottom_cm")),
        )
        return hybras_lookup.get(key, resolve_assignment(row))
    return resolve_assignment(row)


def _hybras_method_lookup(
    workbook_path: Path | None,
) -> dict[tuple[str, str, str], MethodAssignment]:
    if workbook_path is None or not workbook_path.exists():
        return {}
    from carbono_solo.collector.sources.soildata_tabular import _xlsx_sheet_texts

    text = _xlsx_sheet_texts(workbook_path.read_bytes()).get("HYBRAS", "")
    rows = list(csv.reader(text.splitlines(), delimiter="\t"))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if "org_carb (%)" in row and "OC_OM_Method" in row
        ),
        None,
    )
    if header_index is None:
        return {}
    header = list(rows[header_index])
    header[0] = "code"
    lookup = {}
    for values in rows[header_index + 1 :]:
        row = dict(zip(header, values))
        code = _clean_text(row.get("code"))
        top = _number_key(row.get("top_depth"))
        bottom = _number_key(row.get("bot_depth"))
        if code and top and bottom:
            lookup[(f"HYBRAS:{code}", top, bottom)] = hybras_assignment(
                row.get("OC_OM_Method")
            )
    return lookup


def _propagate_unambiguous_duplicate_methods(
    datasets: list[tuple[Path, list[dict[str, str]], list[MethodAssignment]]],
) -> list[tuple[Path, list[dict[str, str]], list[MethodAssignment]]]:
    by_group: dict[str, set[str]] = {}
    for _path, rows, _assignments in datasets:
        for row in rows:
            group_id = _clean_text(row.get("duplicate_group_id"))
            method = _clean_text(row.get("carbon_analysis_method"))
            if group_id and method not in {"", NOT_REPORTED, NOT_APPLICABLE_PSEUDO}:
                by_group.setdefault(group_id, set()).add(method)

    unique_methods = {
        group_id: next(iter(methods))
        for group_id, methods in by_group.items()
        if len(methods) == 1
    }
    result = []
    for path, rows, assignments in datasets:
        mutable_assignments = list(assignments)
        for row_index, row in enumerate(rows):
            if row.get("carbon_analysis_method") != NOT_REPORTED:
                continue
            method = unique_methods.get(_clean_text(row.get("duplicate_group_id")))
            if not method:
                continue
            row["carbon_analysis_method"] = method
            mutable_assignments[row_index] = MethodAssignment(
                method,
                "propagado_por_grupo_duplicado_inequivoco",
                _clean_text(row.get("duplicate_group_id")),
            )
        result.append((path, rows, mutable_assignments))
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _fields_with_method(rows: list[Mapping[str, object]]) -> list[str]:
    if not rows:
        return ["carbon_analysis_method"]
    fields = [field for field in rows[0] if field != "carbon_analysis_method"]
    insert_at = fields.index("carbon_content_g_kg") + 1
    fields.insert(insert_at, "carbon_analysis_method")
    return fields


def _dataset_code(value: object) -> str:
    text = _clean_text(value)
    return text.rsplit("/", 1)[-1].upper()


def _number_key(value: object) -> str:
    text = _clean_text(value).replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return text
    return f"{number:.10f}".rstrip("0").rstrip(".")


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", _clean_text(value))
    ascii_text = text.encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text).split())
