"""Generate the QFT scalability family (rebuilt 2026-11-09).

Textbook QFT without final swaps: for each qubit i, cp(pi/2^(i-j-1)) q[i], q[j] for
every j < i, then h q[i]; a barrier and a measurement of every qubit. Angles below
pi/2^31 are written as cp(0), as in every existing file. Reproduces every
qft_n<N>.qasm byte for byte except qft_n75.qasm, an older variant with no
bit registers and no cp(0) clamp (left as it is).

    python3 benchmark/scalability/qft/generate.py 125 175
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def header(n):
    return ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{n}] c;", f"bit[{n}] meas;", f"qubit[{n}] q;"]
def footer(n):
    return ["barrier " + ", ".join(f"q[{i}]" for i in range(n)) + ";"] + [f"meas[{i}] = measure q[{i}];" for i in range(n)]
def qft(n):
    L = header(n)
    for i in range(n):
        for j in range(i):
            e = i - j - 1
            a = "pi" if e == 0 else (f"pi/{2**e}" if e <= 31 else "0")
            L.append(f"cp({a}) q[{i}], q[{j}];")
        L.append(f"h q[{i}];")
    return "\n".join(L + footer(n)) + "\n"
def adder(n):
    nb = (n - 1) // 9; assert 9 * nb + 1 == n
    A = lambda k, i: 4 * k + i; B = lambda k, i: 4 * nb + 4 * k + i; C = lambda k: 8 * nb + k
    L = header(n)
    xs = [A(k, i) for k in range(nb) for i in (1, 2, 3)] + [B(0, 0), C(0)]
    L += [f"x q[{q}];" for q in sorted(xs)]
    cx = lambda a, b: L.append(f"cx q[{a}], q[{b}];")
    ccx = lambda a, b, c: L.append(f"ccx q[{a}], q[{b}], q[{c}];")
    for k in range(nb):
        cin, cout = C(k), C(k + 1)
        cx(A(k,0), B(k,0)); cx(A(k,0), cin); ccx(cin, B(k,0), A(k,0))
        for i in range(1, 4):
            cx(A(k,i), B(k,i)); cx(A(k,i), A(k,i-1)); ccx(A(k,i-1), B(k,i), A(k,i))
        cx(A(k,3), cout)
        for i in range(3, 0, -1):
            ccx(A(k,i-1), B(k,i), A(k,i)); cx(A(k,i), A(k,i-1)); cx(A(k,i-1), B(k,i))
        ccx(cin, B(k,0), A(k,0)); cx(A(k,0), cin); cx(cin, B(k,0))
    return "\n".join(L + footer(n)) + "\n"

if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        path = os.path.join(HERE, f"qft_n{n}.qasm")
        with open(path, "w") as f:
            f.write(qft(n))
        print("wrote", path)
