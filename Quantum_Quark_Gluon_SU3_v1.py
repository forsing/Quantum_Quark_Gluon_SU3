# Quantum_Quark_Gluon_SU3



import csv
import math
import random
from itertools import combinations, islice
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Operator, Statevector


random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4720_k91.csv"),
    Path("/data/loto7_4720_k91_loto_2982.csv"),
    Path("/data/loto7_4720_k91_loto_plus_1738.csv"),
)



NUMBERS = 39
PICKS = 7
COLOR_DIM = 3
NUMBER_QUBITS = 6
MAX_NEXT_RUN = 4

TRAIN_BATCH = 512
SEARCH_BATCH = 4096
REG_GRID = (0.001, 0.01, 0.1)
GRAD_TOL = 0.001

OPTIMIZATION_STAGES = (
    ("BFGS", 250),
    ("CG", 350),
    ("BFGS", 350),
)

IDENTITY3 = np.eye(3, dtype=complex)


class ConvergenceError(RuntimeError):
    pass


def adjoint(matrix):
    return matrix.conj().swapaxes(-1, -2)


def make_generators():
    """Osam Gell-Mann generatora, sa konvencijom T_a=lambda_a/2."""
    result = []

    for first, second in ((0, 1), (0, 2), (1, 2)):
        symmetric = np.zeros((3, 3), dtype=complex)
        symmetric[first, second] = 0.5
        symmetric[second, first] = 0.5
        result.append(symmetric)

        antisymmetric = np.zeros((3, 3), dtype=complex)
        antisymmetric[first, second] = -0.5j
        antisymmetric[second, first] = 0.5j
        result.append(antisymmetric)

    result.append(np.diag([0.5, -0.5, 0.0]).astype(complex))
    result.append(
        np.diag([1.0, 1.0, -2.0]).astype(complex)
        / (2.0 * np.sqrt(3.0))
    )

    generators = np.stack(result)

    if not np.allclose(generators, adjoint(generators), atol=1e-13):
        raise RuntimeError("SU(3) generator nije hermitski.")

    if not np.allclose(
        np.trace(generators, axis1=1, axis2=2), 0.0, atol=1e-13
    ):
        raise RuntimeError("SU(3) generator nema nulti trag.")

    return generators


GENERATORS = make_generators()
PARAM_COUNT = NUMBERS * len(GENERATORS)


def initial_parameters():
    number = np.arange(1, NUMBERS + 1, dtype=float)[:, None]
    generator = np.arange(1, len(GENERATORS) + 1, dtype=float)[None, :]

    angles = 1.1 * np.sin(
        np.sqrt(2.0) * number
        + np.sqrt(3.0) * generator
        + np.sqrt(5.0) * number * generator
    )

    return angles.ravel()


def build_links(params, derivatives=False):
    """
    Za svaki broj postoji SU(3) operator:
    U_n = exp(i sum_a theta[n,a] T_a).

    Parametri su koeficijenti kvantnih generatora.
    Nisu regresione tezine izlaznog sloja.
    """
    angles = np.asarray(params).reshape(NUMBERS, len(GENERATORS))
    h = np.einsum("na,aij->nij", angles, GENERATORS, optimize=True)

    eigenvalues, eigenvectors = np.linalg.eigh(h)
    phases = np.exp(1j * eigenvalues)
    links = (
        eigenvectors * phases[:, None, :]
    ) @ adjoint(eigenvectors)

    if not derivatives:
        return links

    difference = eigenvalues[:, :, None] - eigenvalues[:, None, :]
    midpoint = (
        eigenvalues[:, :, None] + eigenvalues[:, None, :]
    ) / 2.0

    divided_difference = (
        1j
        * np.exp(1j * midpoint)
        * np.sinc(difference / (2.0 * np.pi))
    )

    transformed = (
        adjoint(eigenvectors)[:, None, :, :]
        @ GENERATORS[None, :, :, :]
        @ eigenvectors[:, None, :, :]
    )

    derivative_links = (
        eigenvectors[:, None, :, :]
        @ (transformed * divided_difference[:, None, :, :])
        @ adjoint(eigenvectors)[:, None, :, :]
    )

    return links, derivative_links


def load_draws(path):
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle), 1):
            if not fields or all(not item.strip() for item in fields):
                continue

            if len(fields) != PICKS:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "potrebno je tacno sedam kolona bez hedera."
                )

            try:
                values = [int(item.strip()) for item in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}, red {line_number}: neispravan broj."
                ) from exc

            if (
                len(set(values)) != PICKS
                or min(values) < 1
                or max(values) > NUMBERS
            ):
                raise ValueError(
                    f"{path.name}, red {line_number}: nelegalna kombinacija."
                )

            rows.append(sorted(values))

    if len(rows) < 80:
        raise ValueError(f"{path.name}: nedovoljno istorijskih kola.")

    # Hronologija ostaje nepromenjena.
    return np.asarray(rows, dtype=np.int64) - 1


def chain_forward(links, sequences, keep_tape=False):
    """P = U_last ... U_second U_first."""
    states = np.broadcast_to(
        IDENTITY3, (len(sequences), 3, 3)
    ).copy()

    tape = []

    for position in range(sequences.shape[1]):
        tokens = sequences[:, position]

        if keep_tape:
            tape.append(states)

        states = links[tokens] @ states

    return states, tape


def chain_backward(links, sequences, tape, gradient):
    link_gradient = np.zeros_like(links)
    current = gradient.copy()

    for position in range(sequences.shape[1] - 1, -1, -1):
        tokens = sequences[:, position]

        increments = current @ adjoint(tape[position])
        np.add.at(link_gradient, tokens, increments)

        current = adjoint(links[tokens]) @ current

    return link_gradient


def lift_link(link):
    """
    Kolonska vektorizacija:
    vec(U P) = (I kron U) vec(P).
    """
    return np.kron(IDENTITY3, link)


def gram_forward(links, picks, max_run, keep_tape=False):
    """
    G = sum_S vec(P_S) vec(P_S)^dagger.

    Dinamicko programiranje po:
    - broju odabranih elemenata,
    - duzini trenutnog niza uzastopnih elemenata.

    max_run=picks ukljucuje sve kombinacije.
    """
    vector_identity = IDENTITY3.reshape(9, order="F")

    state = np.zeros(
        (picks + 1, max_run + 1, 9, 9), dtype=complex
    )
    state[0, 0] = np.outer(
        vector_identity, vector_identity.conj()
    )

    tape = []

    for link in links:
        if keep_tape:
            tape.append(state)

        lifted = lift_link(link)
        updated = np.zeros_like(state)

        # Preskakanje broja prekida tekuci uzastopni niz.
        updated[:, 0] = state.sum(axis=1)

        # Uzimanje broja produzuje niz za jedan.
        updated[1:, 1:] = (
            lifted
            @ state[:-1, :max_run]
            @ adjoint(lifted)
        )

        state = updated

    gram = state[picks].sum(axis=0)
    gram = 0.5 * (gram + adjoint(gram))
    return gram, tape


def gram_backward(links, tape, picks, max_run, gradient):
    state_gradient = np.zeros(
        (picks + 1, max_run + 1, 9, 9), dtype=complex
    )

    state_gradient[picks, :] = gradient
    link_gradient = np.zeros_like(links)

    for number in range(len(links) - 1, -1, -1):
        previous = tape[number]
        lifted = lift_link(links[number])

        # Izvod grane "preskoci".
        previous_gradient = np.broadcast_to(
            state_gradient[:, 0, None, :, :],
            previous.shape,
        ).copy()

        selected_gradient = state_gradient[1:, 1:]
        selected_previous = previous[:-1, :max_run]

        # Izvod grane "uzmi".
        previous_gradient[:-1, :max_run] += (
            adjoint(lifted)
            @ selected_gradient
            @ lifted
        )

        lifted_gradient = (
            (selected_gradient + adjoint(selected_gradient))
            @ lifted
            @ selected_previous
        ).sum(axis=(0, 1))

        # lifted = I_3 kron link.
        for block in range(3):
            first = 3 * block
            link_gradient[number] += lifted_gradient[
                first:first + 3, first:first + 3
            ]

        state_gradient = previous_gradient

    return link_gradient


def normalization(contexts, gram):
    # trace(P C) = vec(C.T)^T vec(P).
    # vec(C.T), uz kolonsku vektorizaciju, jeste C.ravel().
    vectors = contexts.reshape(len(contexts), 9)

    z = np.einsum(
        "bi,ij,bj->b",
        vectors,
        gram,
        vectors.conj(),
        optimize=True,
    ).real

    if np.any(~np.isfinite(z)) or np.any(z <= 0.0):
        raise FloatingPointError("Neispravna normalizaciona konstanta.")

    return vectors, z


def objective(params, draws, start, end, regularization, gradient=True):
    count = end - start

    if start < 1 or count <= 0:
        raise ValueError("Neispravan opseg ucenja.")

    if gradient:
        links, derivative_links = build_links(params, True)
    else:
        links = build_links(params)

    # Ucenje i test ukljucuju SVA istorijska kola.
    # Ogranicenje max cetiri primenjuje se pri NEXT izboru.
    gram, gram_tape = gram_forward(
        links, PICKS, PICKS, gradient
    )

    loss = 0.0
    link_gradient = np.zeros_like(links)
    gram_gradient = np.zeros((9, 9), dtype=complex)

    for first in range(start, end, TRAIN_BATCH):
        last = min(first + TRAIN_BATCH, end)

        histories = draws[first - 1:last - 1]
        targets = draws[first:last]

        contexts, context_tape = chain_forward(
            links, histories, gradient
        )
        products, product_tape = chain_forward(
            links, targets, gradient
        )

        amplitudes = np.einsum(
            "bij,bji->b", products, contexts, optimize=True
        )
        absolute = np.abs(amplitudes)

        if np.any(absolute == 0.0):
            if gradient:
                return np.inf, np.zeros_like(params)
            return np.inf

        vectors, z = normalization(contexts, gram)

        loss += float(
            (np.log(z) - 2.0 * np.log(absolute)).sum()
        )

        if gradient:
            denominator = amplitudes.conj()[:, None, None]

            product_gradient = -adjoint(contexts) / denominator
            context_gradient = -adjoint(products) / denominator

            context_gradient += (
                (vectors @ gram) / z[:, None]
            ).reshape(-1, 3, 3)

            gram_gradient += 0.5 * np.einsum(
                "bi,bj,b->ij",
                vectors.conj(),
                vectors,
                1.0 / z,
                optimize=True,
            )

            link_gradient += chain_backward(
                links, targets, product_tape, product_gradient
            )
            link_gradient += chain_backward(
                links, histories, context_tape, context_gradient
            )

    loss /= count
    loss += regularization * float(np.mean(params ** 2))

    if not gradient:
        return loss

    link_gradient += gram_backward(
        links, gram_tape, PICKS, PICKS, gram_gradient
    )
    link_gradient /= count

    parameter_gradient = 2.0 * np.real(
        np.einsum(
            "nij,naij->na",
            link_gradient.conj(),
            derivative_links,
            optimize=True,
        )
    ).ravel()

    parameter_gradient += (
        2.0 * regularization * params / len(params)
    )

    if (
        not np.isfinite(loss)
        or not np.all(np.isfinite(parameter_gradient))
    ):
        raise FloatingPointError("Nefinitan cilj ili gradijent.")

    return loss, parameter_gradient


def fit(draws, end, regularization, initial=None, label="ucenje"):
    params = (
        initial_parameters()
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )

    best_x = None
    best_value = np.inf
    best_gradient = None

    cache_x = None
    cache_value = None
    cache_gradient = None

    def function(values):
        nonlocal best_x, best_value, best_gradient
        nonlocal cache_x, cache_value, cache_gradient

        values = np.asarray(values, dtype=float)

        if cache_x is not None and np.array_equal(values, cache_x):
            return cache_value, cache_gradient.copy()

        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Nefinitni parametri.")

        value, derivative = objective(
            values, draws, 1, end, regularization, True
        )

        cache_x = values.copy()
        cache_value = float(value)
        cache_gradient = derivative.copy()

        if np.isfinite(value):
            better = value < best_value

            if best_gradient is not None and value == best_value:
                better = (
                    np.max(np.abs(derivative))
                    < np.max(np.abs(best_gradient))
                )

            if better:
                best_x = values.copy()
                best_value = float(value)
                best_gradient = derivative.copy()

        return float(value), derivative.copy()

    initial_value, initial_gradient = function(params)

    if not np.isfinite(initial_value):
        raise FloatingPointError(
            f"{label}: nulta amplituda posmatranog prelaza."
        )

    if np.max(np.abs(initial_gradient)) <= GRAD_TOL:
        return params

    reports = []

    for stage, (method, limit) in enumerate(OPTIMIZATION_STAGES):
        if stage == 1:
            print(
                f"Nastavak optimizacije: {label}; "
                f"gradijent={np.max(np.abs(best_gradient)):.6g}.",
                flush=True,
            )

        result = minimize(
            function,
            best_x.copy(),
            jac=True,
            method=method,
            options={
                "maxiter": limit,
                "gtol": GRAD_TOL,
                "norm": np.inf,
            },
        )

        value, derivative = function(result.x)
        norm = float(np.max(np.abs(derivative)))

        reports.append(
            f"{method}: status={result.status}, "
            f"iteracije={result.nit}, gradijent={norm:.6g}, "
            f"razlog={str(result.message).strip()}"
        )

        if (
            np.isfinite(value)
            and norm <= GRAD_TOL
            and value <= initial_value + 1e-10
        ):
            return np.asarray(result.x).copy()

        if (
            np.max(np.abs(best_gradient)) <= GRAD_TOL
            and best_value <= initial_value + 1e-10
        ):
            return best_x.copy()

    raise ConvergenceError(
        f"{label}: konvergencija nije dostignuta. "
        + " | ".join(reports)
    )


def allowed_rows(sequences, max_run):
    if sequences.shape[1] <= max_run:
        return np.ones(len(sequences), dtype=bool)

    valid = np.ones(len(sequences), dtype=bool)

    for first in range(sequences.shape[1] - max_run):
        # Za strogo rastuce cele brojeve ovaj uslov znaci
        # da postoji max_run+1 uzastopnih brojeva.
        valid &= (
            sequences[:, first + max_run] - sequences[:, first]
            > max_run
        )

    return valid


def exhaustive_next(links, context, picks, max_run, batch_size):
    gram, _ = gram_forward(links, picks, max_run, False)
    _, z = normalization(context[None, :, :], gram)
    z = float(z[0])

    iterator = combinations(range(len(links)), picks)
    best_weight = -1.0
    best_sequence = None

    masses = []
    visited = 0
    accepted = 0

    while True:
        chunk = list(islice(iterator, batch_size))

        if not chunk:
            break

        sequences = np.asarray(chunk, dtype=np.int64)
        visited += len(sequences)

        valid = allowed_rows(sequences, max_run)
        sequences = sequences[valid]
        accepted += len(sequences)

        if not len(sequences):
            continue

        products, _ = chain_forward(links, sequences, False)
        amplitudes = np.einsum(
            "bij,ji->b", products, context, optimize=True
        )
        weights = np.abs(amplitudes) ** 2

        if not np.all(np.isfinite(weights)):
            raise FloatingPointError("Nefinitna NEXT tezina.")

        masses.append(float(weights.sum()))

        index = int(np.argmax(weights))
        weight = float(weights[index])

        if weight > best_weight:
            best_weight = weight
            best_sequence = tuple(
                int(number) for number in sequences[index]
            )

    if visited != math.comb(len(links), picks):
        raise RuntimeError("Pretraga nije obuhvatila sve kombinacije.")

    enumerated_z = math.fsum(masses)

    if not np.isclose(enumerated_z, z, atol=1e-8, rtol=1e-8):
        raise FloatingPointError(
            "Enumeracija i dinamicka normalizacija se ne slazu."
        )

    if best_sequence is None or best_weight <= 0.0:
        raise RuntimeError("Nema dozvoljene kombinacije pozitivne mase.")

    return best_sequence, best_weight / z, accepted


def embed_color(matrix):
    embedded = np.eye(4, dtype=complex)
    embedded[:3, :3] = matrix
    return embedded


def number_color_gate(links):
    """
    Sest qubita oznake broja + dva qubita boje.
    Boje 0,1,2 su fizicki SU(3) potprostor.
    """
    matrix = np.eye(256, dtype=complex)

    for number, link in enumerate(links, start=1):
        indices = number + 64 * np.arange(3)
        matrix[np.ix_(indices, indices)] = link

    return UnitaryGate(matrix, label="SU3(number)")


def build_qiskit_circuit(links, context):
    """
    Sedam izlaznih registara po sest qubita i cetiri pomocna qubita.

    Petlja se zatvara projekcijom pomocnih registara na |0000>.
    Legalan rastuci izlaz i ogranicenje niza predstavljaju dodatni
    uslov na izlaz. Numericki kod racuna tu uslovnu raspodelu tacno.

    Ne simulira se puni statevector ovog 46-qubitnog kola.
    """
    outputs = [
        QuantumRegister(NUMBER_QUBITS, f"n{position + 1}")
        for position in range(PICKS)
    ]
    color = QuantumRegister(2, "color")
    reference = QuantumRegister(2, "reference")

    circuit = QuantumCircuit(*outputs, color, reference)

    number_state = np.zeros(64, dtype=complex)
    number_state[1:NUMBERS + 1] = 1.0 / np.sqrt(NUMBERS)
    prepare_number = StatePreparation(number_state)

    # Maksimalno spregnut par trostanjskih sistema.
    bell_state = np.zeros(16, dtype=complex)
    bell_state[[0, 5, 10]] = 1.0 / np.sqrt(3.0)
    prepare_bell = StatePreparation(bell_state)

    auxiliary = list(color) + list(reference)
    circuit.append(prepare_bell, auxiliary)
    circuit.append(UnitaryGate(embed_color(context)), list(color))

    controlled = number_color_gate(links)

    for register in outputs:
        circuit.append(prepare_number, list(register))
        circuit.append(controlled, list(register) + list(color))

    circuit.append(prepare_bell.inverse(), auxiliary)
    return circuit


def self_check():
    params = initial_parameters()
    links, derivatives = build_links(params, True)

    if not np.allclose(
        adjoint(links) @ links, IDENTITY3, atol=1e-11, rtol=1e-11
    ):
        raise RuntimeError("SU(3) operatori nisu unitarni.")

    if not np.allclose(
        np.linalg.det(links), 1.0, atol=1e-11, rtol=1e-11
    ):
        raise RuntimeError("SU(3) determinanta nije jedan.")

    # Provera izvoda SU(3) eksponencijala.
    step = 1e-6

    for number, generator in ((0, 0), (11, 3), (38, 7)):
        index = 8 * number + generator
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step

        numerical = (
            build_links(plus)[number] - build_links(minus)[number]
        ) / (2.0 * step)

        if not np.allclose(
            numerical,
            derivatives[number, generator],
            atol=2e-7,
            rtol=2e-5,
        ):
            raise RuntimeError("Izvod SU(3) operatora nije ispravan.")

    # Mala iscrpna provera Gramove normalizacije i maksimuma.
    small_links = links[:6]
    context = links[8] @ links[3]

    for picks, max_run in ((1, 1), (3, 3), (3, 2)):
        selected, probability, _ = exhaustive_next(
            small_links, context, picks, max_run, 4
        )

        scores = []
        sequences = []

        for sequence in combinations(range(len(small_links)), picks):
            row = np.asarray([sequence], dtype=np.int64)

            if not allowed_rows(row, max_run)[0]:
                continue

            product = IDENTITY3.copy()

            for number in sequence:
                product = small_links[number] @ product

            sequences.append(sequence)
            scores.append(float(abs(np.trace(product @ context)) ** 2))

        scores = np.asarray(scores)
        expected = float(scores.max() / scores.sum())
        selected_index = sequences.index(selected)

        if not np.isclose(
            probability, expected, atol=1e-11, rtol=1e-10
        ):
            raise RuntimeError("Provera MAP maksimuma nije prosla.")

        if not np.isclose(
            scores[selected_index], scores.max(), atol=1e-11
        ):
            raise RuntimeError("Izabrana putanja nije maksimum.")

    # Posebna provera pravila 20.
    examples = np.asarray(
        [
            [0, 1, 2, 3, 4, 10, 20],
            [0, 1, 2, 3, 8, 9, 10],
            [0, 1, 2, 3, 4, 5, 6],
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        allowed_rows(examples, 4),
        np.asarray([False, True, False]),
    ):
        raise RuntimeError("Provera ogranicenja uzastopnih brojeva nije prosla.")

    # Qiskit provera operatora uslovljenog brojem.
    gate = number_color_gate(links)

    for number in (1, 19, 39):
        for color in (0, 1, 2):
            initial_index = number + 64 * color
            result = Statevector.from_int(
                initial_index, dims=256
            ).evolve(gate)

            expected = np.zeros(256, dtype=complex)
            expected[number + 64 * np.arange(3)] = links[number - 1][:, color]

            if not np.allclose(result.data, expected, atol=1e-11):
                raise RuntimeError("Qiskit provera prenosa boje nije prosla.")

    # Trag proizvoda kao koherentna amplituda zatvorene petlje.
    bell = np.zeros(16, dtype=complex)
    bell[[0, 5, 10]] = 1.0 / np.sqrt(3.0)
    preparation = StatePreparation(bell)

    small_circuit = QuantumCircuit(4)
    small_circuit.append(preparation, range(4))
    small_circuit.append(UnitaryGate(embed_color(context)), [0, 1])

    product = IDENTITY3.copy()

    for number in (2, 7, 12):
        small_circuit.append(
            UnitaryGate(embed_color(links[number])), [0, 1]
        )
        product = links[number] @ product

    small_circuit.append(preparation.inverse(), range(4))
    amplitude = Statevector.from_instruction(small_circuit).data[0]

    if not np.allclose(
        amplitude, np.trace(product @ context) / 3.0, atol=1e-11
    ):
        raise RuntimeError("Qiskit provera amplitude petlje nije prosla.")

    # Fiksni testni podaci, ne generisanje NEXT kombinacija.
    sample = np.asarray(
        [
            [1, 4, 9, 16, 22, 31, 38],
            [2, 7, 11, 18, 25, 30, 39],
            [3, 5, 12, 20, 24, 32, 37],
            [4, 8, 14, 19, 27, 33, 36],
        ],
        dtype=np.int64,
    ) - 1

    _, analytical = objective(
        params, sample, 1, len(sample), 0.01, True
    )

    directions = []

    for index in (0, 7, 8, 79, 151, PARAM_COUNT - 1):
        direction = np.zeros(PARAM_COUNT)
        direction[index] = 1.0
        directions.append(direction)

    coordinates = np.arange(1, PARAM_COUNT + 1, dtype=float)

    for frequency in (0.37, 0.83, 1.41):
        direction = np.sin(frequency * coordinates)
        direction /= np.linalg.norm(direction)
        directions.append(direction)

    for direction in directions:
        upper = objective(
            params + step * direction,
            sample, 1, len(sample), 0.01, False,
        )
        lower = objective(
            params - step * direction,
            sample, 1, len(sample), 0.01, False,
        )

        numerical = (upper - lower) / (2.0 * step)
        predicted = float(np.dot(analytical, direction))

        if not np.isclose(
            numerical, predicted, atol=3e-5, rtol=3e-4
        ):
            raise RuntimeError(
                "Provera gradijenta nije prosla: "
                f"analiticki={predicted:.9g}, numericki={numerical:.9g}."
            )


def process_csv(path):
    draws = load_draws(path)
    count = len(draws) - 1

    train_end = 1 + int(0.70 * count)
    validation_end = 1 + int(0.85 * count)
    best = None

    for regularization in REG_GRID:
        try:
            parameters = fit(
                draws,
                train_end,
                regularization,
                label=f"trening reg={regularization:g}",
            )
        except ConvergenceError as exc:
            print(f"PROVERA: {exc}", flush=True)
            continue

        validation = objective(
            parameters,
            draws,
            train_end,
            validation_end,
            0.0,
            False,
        )

        if np.isfinite(validation):
            if best is None or validation < best[0]:
                best = (validation, regularization, parameters)

    if best is None:
        raise ConvergenceError("Nema konvergiranog validacionog kandidata.")

    validation, regularization, train_parameters = best

    test_parameters = fit(
        draws,
        validation_end,
        regularization,
        initial=train_parameters,
        label="ucenje trening + validacija",
    )

    test = objective(
        test_parameters,
        draws,
        validation_end,
        len(draws),
        0.0,
        False,
    )

    full_parameters = fit(
        draws,
        len(draws),
        regularization,
        initial=test_parameters,
        label="zavrsno ucenje na celoj istoriji",
    )

    links = build_links(full_parameters)
    contexts, _ = chain_forward(links, draws[-1:], False)
    context = contexts[0]

    # Konstrukcija kola, bez formiranja njegovog punog statevectora.
    circuit = build_qiskit_circuit(links, context)

    if circuit.num_qubits != PICKS * NUMBER_QUBITS + 4:
        raise RuntimeError("Neispravna struktura kvantnog kola.")

    print(
        f"Validacija NLL: {validation:.6f} | reg: {regularization:g}",
        flush=True,
    )
    print(f"Test NLL: {test:.6f}", flush=True)
    print(
        "Pretraga NEXT: iscrpna, najvise cetiri uzastopna broja.",
        flush=True,
    )

    selected, probability, _ = exhaustive_next(
        links, context, PICKS, MAX_NEXT_RUN, SEARCH_BATCH
    )

    next_combination = tuple(number + 1 for number in selected)
    print(f"NEXT: {list(next_combination)}", flush=True)

    if not np.isfinite(test):
        print(
            "PROVERA: najmanje jedan test prelaz ima nultu amplitudu.",
            flush=True,
        )

    return {
        "path": path,
        "next": next_combination,
        "parameters": full_parameters,
        "probability": probability,
    }


def main():
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(str(path))

    print(
        "Quantum Quark-Gluon SU(3): "
        "neabelovski prenos boje i zajednicka amplituda kombinacije.",
        flush=True,
    )
    print(
        "Lokalna sazetа simulacija; "
        "sedam brojnih registara i pomocni kolorni registri.",
        flush=True,
    )

    self_check()
    print("Interne provere: prosle.", flush=True)

    results = []

    for path in FILES:
        print(f"CSV: {path.name}", flush=True)

        try:
            results.append(process_csv(path))
        except ConvergenceError as exc:
            print(f"STOP za ovaj CSV: {exc}", flush=True)

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            left = results[first]
            right = results[second]

            if left["next"] == right["next"]:
                same_parameters = np.allclose(
                    left["parameters"],
                    right["parameters"],
                    atol=1e-10,
                    rtol=1e-8,
                )

                print(
                    "PROVERA: ista NEXT kombinacija za "
                    f"{left['path'].name} i {right['path'].name}.",
                    flush=True,
                )
                print(
                    "Parametri modela: "
                    + (
                        "prakticno isti."
                        if same_parameters
                        else "razliciti."
                    ),
                    flush=True,
                )

    if len(results) != len(FILES):
        print(
            "Obrada nije kompletna: neki modeli nisu konvergirali.",
            flush=True,
        )


if __name__ == "__main__":
    main()



"""
Quantum Quark-Gluon SU(3): neabelovski prenos boje i zajednicka amplituda kombinacije.
Lokalna sazeta simulacija; sedam brojnih registara i pomocni kolorni registri.
Interne provere: prosle.

CSV: loto7_4720_k91.csv
Validacija NLL: 17.076520 | reg: 0.001
Test NLL: 17.172954
Pretraga NEXT: iscrpna, najvise cetiri uzastopna broja.
NEXT: [9, x, 17, y, 25, z, 38]

CSV: loto7_4720_k91_loto_2982.csv
Validacija NLL: 17.021851 | reg: 0.001
Test NLL: 17.074312
Pretraga NEXT: iscrpna, najvise cetiri uzastopna broja.
NEXT: [1, x, 11, y, 23, z, 38]

CSV: loto7_4720_k91_loto_plus_1738.csv
Nastavak optimizacije: trening reg=0.01; gradijent=0.00221162.
Validacija NLL: 17.024549 | reg: 0.01
Test NLL: 17.164463
Pretraga NEXT: iscrpna, najvise cetiri uzastopna broja.
NEXT: [11, x, 19, y, 25, z, 33]
"""



"""
Kvarkovi i gluoni	
Kvarkovi imaju unutrašnji „kolorni” stepen slobode. Gluoni prenose tu interakciju i međusobno interaguju.	
Odnos između dva elementa zavisi i od ostalih, umesto nezavisnog ocenjivanja pojedinačnih brojeva.


Kvark-gluonske interakcije: 
- Brojevi ostaju oznake 1-39, bez nametnutog numeričkog susedstva.
- Unutrašnja kvantna stanja predstavljaju skrivene odnose naučene iz istorije, a ne unapred određene grupe brojeva.
- Interakcije obuhvataju više članova kombinacije, pa se uči zajednička amplituda svih sedam brojeva.
- NEXT se bira prema toj zajedničkoj raspodeli, uz moje ograničenje od najviše četiri uzastopna broja.

Mogućnost učenja višestrukih međuzavisnosti u celoj kombinaciji. 


Model koristi osam SU(3) generatora i uređene proizvode operatora za zajedničku amplitudu kombinacije. 
Osnova su neabelovski prenos boje i trag proizvoda operatora, inspirisan Wilsonovim petljama. 

Adaptacija za učenje raspodele, ne simulacija pune QCD niti dokaz prediktivne prednosti.
- Svaki CSV se uči zasebno.
- Sva istorijska kola ostaju u učenju.
- NEXT pretraga dozvoljava najviše četiri uzastopna broja.
- Nema slučajnog generisanja kombinacija.
- Uključene su provere gradijenta, normalizacije, maksimuma i Qiskit operacija.
"""
