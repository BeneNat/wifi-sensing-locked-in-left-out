# Passive CSI collection

Listens passively for packets on channel 3 (same channel as both active_ap and active_sta). This channel can be changed in `main/main.c` depending on the channel of the device you wish to passively listen for.

The easiest way to evaluate this sub-project is to flash three ESP32s. One with active_ap, one with active_sta and finally one with this passive sub-project.

To use run `idf.py flash monitor` from a terminal.

---

## Attribution and Origin

This component is based on the original implementation provided by the **[ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)** repository, developed by Steven M. Hernandez. 

It has been integrated into the **LILO** (Locked-In Left-Out) project as the fundamental layer for Channel State Information (CSI) data acquisition. The tool's core logic for CSI initialization and packet capturing remains consistent with the original repository to ensure compatibility with standard CSI processing workflows.

*This integration is part of an Engineering Thesis project at the **AGH University of Krakow**.*