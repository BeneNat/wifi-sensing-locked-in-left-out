# Active CSI collection (Station)

Connects to some Access Point (AP) (Router or another ESP32) and sends packet requests (thus receiving CSI packet responses).

To use run `idf.py flash monitor` from a terminal.

This sub-project most commonly pairs with the project in `./active_ap`. Flash these two sub-projects to two different ESP32s to quickly begin collecting Channel State Information.

---

## Attribution and Origin

This component is based on the original implementation provided by the **[ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)** repository, developed by Steven M. Hernandez. 

It has been integrated into the **LILO** (Locked-In Left-Out) project as the fundamental layer for Channel State Information (CSI) data acquisition. The tool's core logic for CSI initialization and packet capturing remains consistent with the original repository to ensure compatibility with standard CSI processing workflows.

*This integration is part of an Engineering Thesis project at the **AGH University of Krakow**.*