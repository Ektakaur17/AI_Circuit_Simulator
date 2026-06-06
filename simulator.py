import subprocess
import os
import re

print("=== AI Circuit Simulator ===")

# User input
V = float(input("Enter Supply Voltage (V): "))
R1 = float(input("Enter R1 value (Ohms): "))
R2 = float(input("Enter R2 value (Ohms): "))

# Create netlist dynamically
netlist = f"""
* Voltage Divider
V1 in 0 DC {V}
R1 in mid {R1}
R2 mid 0 {R2}
.op
.end
"""

with open("auto.sp", "w") as f:
    f.write(netlist)

print("\nNetlist file created")

ngspice_path = r"C:\Users\Harsh Belwal\Downloads\ngspice-45.2_64\Spice64\bin\ngspice.exe"

subprocess.run(
    [ngspice_path, "-b", "auto.sp", "-o", "result.txt"]
)

print("Simulation completed")

if os.path.exists("result.txt"):
    with open("result.txt", "r") as f:
        data = f.read()

    match = re.search(r"mid\s+([0-9.eE+-]+)", data)
    if match:
        voltage = float(match.group(1))
        print("\n✅ Output Voltage at mid node =", voltage, "Volts")
    else:
        print("Voltage not found")

print("\nProgram Finished")
