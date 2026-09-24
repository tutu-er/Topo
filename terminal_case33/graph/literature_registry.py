"""Machine-readable registry of implemented topology-identification baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LiteratureMethod:
    """Citation, implementation scope, and observability assumptions."""

    key: str
    citation: str
    doi_or_url: str
    implementation: str
    fidelity: str
    measurements: str
    assumptions: str
    output_topology: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable record."""

        return asdict(self)


LITERATURE_METHODS: tuple[LiteratureMethod, ...] = (
    LiteratureMethod(
        "soumalas2017_prufer",
        "Soumalas, Messinis, and Hatziargyriou, IEEE Manchester PowerTech, 2017",
        "https://doi.org/10.1109/PTC.2017.7981061",
        "terminal_case33.graph.prufer.soumalas_prufer_reconstruction",
        "paper-assumption-faithful open implementation; additive-tree solver replaces the paper flowchart",
        "synchronized terminal P/Q/V and known or regulated root",
        "homogeneous candidate line types; lengths are integer multiples of a minimum span",
        "full unit-edge latent tree including degree-2 span nodes",
    ),
    LiteratureMethod(
        "park2020_recursive_grouping",
        "Park, Deka, Backhaus, and Chertkov, IEEE TCNS, 2020",
        "https://doi.org/10.1109/TCNS.2020.2979882",
        "terminal_case33.graph.latent_tree.recursive_grouping",
        "faithful distance-domain implementation of the joint topology/impedance algorithm",
        "time-aligned terminal voltage and complex-power samples",
        "zero-injection hidden nodes; exact guarantee needs hidden degree at least three and additive distances",
        "minimal latent tree; degree-2 chains are suppressed",
    ),
    LiteratureMethod(
        "pengwah2022_backtracking_rg",
        "Pengwah, Fang, Razzaghi, and Andrew, IEEE Systems Journal, 2022",
        "https://doi.org/10.1109/JSYST.2021.3128175",
        "terminal_case33.graph.enhanced_recursive_grouping.enhanced_recursive_grouping",
        "algorithm-faithful backtracking; private logistic weights replaced by explicit normalized weights",
        "terminal voltage/current/power-factor samples",
        "constant transformer voltage in the original estimator; zero-injection hidden nodes",
        "candidate Steiner trees ranked by four smart-meter objectives",
    ),
    LiteratureMethod(
        "flynn2023_improved_rg",
        "Flynn, Pengwah, Razzaghi, and Andrew, IEEE TSG, 2023",
        "https://doi.org/10.1109/TSG.2023.3239650",
        "terminal_case33.estimation.literature_sensitivity.fit_current_sensitivity plus enhanced_recursive_grouping",
        "algorithm-faithful physical pairing constraints and regularized common transformer mode",
        "terminal voltage/current/power-factor samples; transformer voltage need not be metered",
        "positive line lengths; two metered customers cannot be parent/child; zero-injection hidden nodes",
        "physically feasible candidate Steiner trees",
    ),
    LiteratureMethod(
        "pengwah2024_partial_meter",
        "Pengwah et al., IEEE Transactions on Power Delivery, 2024",
        "https://doi.org/10.1109/TPWRD.2024.3354292",
        "fit_partial_meter_impedance plus insert_interval_meters",
        "paper-faithful two-stage smart/interval-meter implementation",
        "smart P/Q/V; interval active-energy averages; transformer voltage",
        "interval-meter Q and V unavailable; interval power factor approximated as unity",
        "smart-meter reduced tree followed by interval-meter placement",
    ),
    LiteratureMethod(
        "choi2011_clgrouping",
        "Choi, Tan, Anandkumar, and Willsky, JMLR, 2011",
        "https://www.jmlr.org/papers/v12/choi11b.html",
        "terminal_case33.graph.cl_grouping.cl_grouping",
        "distance-tree specialization of published CLGrouping",
        "an additive or information-distance matrix on observed nodes",
        "minimal latent tree and reliable local distances",
        "unrooted minimal latent tree",
    ),
    LiteratureMethod(
        "ni2011_rooted_neighbor_joining",
        "Ni and Tatikonda, IEEE Transactions on Information Theory, 2011",
        "https://doi.org/10.1109/TIT.2011.2168901",
        "terminal_case33.graph.rooted_neighbor_joining.rooted_neighbor_joining",
        "shared-path rooted-neighbor-joining implementation",
        "root-to-terminal and terminal-to-terminal additive distances",
        "known root; positive lower bound on edge lengths for noisy guarantees",
        "rooted minimal latent tree",
    ),
)


def literature_method_records() -> list[dict[str, str]]:
    """Return all method metadata for reports and JSON outputs."""

    return [method.to_dict() for method in LITERATURE_METHODS]
