using PauliPropagation

# We create two identical Pauli Strings
# Y on Qubit 1
p1 = PauliString(1, :Y, 1, 1.0 + 0.0im)
p2 = PauliString(1, :Y, 1, 1.0 + 0.0im)

# We want to apply an S gate in the Schrödinger picture to both strings sequentially.
# For Y, S in Schrödinger picture transforms it to -X
# For Y, S in Heisenberg picture transforms it to +X

# FIRST APPLICATION (correctly uses Schrödinger)
gate_s1 = PauliPropagation.toschrodinger(CliffordGate(:S, [1]))
lookup1 = PauliPropagation.clifford_map[gate_s1.symbol]
res1 = PauliPropagation.PropagationBase.apply(gate_s1, p1.term, p1.coeff, lookup1)
p1_evolved = PauliString(1, res1[1][1], res1[1][2])
println("First S application:  ", p1_evolved)

# SECOND APPLICATION (bug causes it to use Heisenberg)
gate_s2 = PauliPropagation.toschrodinger(CliffordGate(:S, [1]))
lookup2 = PauliPropagation.clifford_map[gate_s2.symbol]
res2 = PauliPropagation.PropagationBase.apply(gate_s2, p2.term, p2.coeff, lookup2)
p2_evolved = PauliString(1, res2[1][1], res2[1][2])
println("Second S application: ", p2_evolved)

if p1_evolved.coeff != p2_evolved.coeff
    println("\n🚨 BUG DETECTED: The second application returned a different sign!")
end
