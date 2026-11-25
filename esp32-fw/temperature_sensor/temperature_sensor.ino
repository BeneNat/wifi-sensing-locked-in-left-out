#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>

#ifndef LED_BUILTIN
#define LED_BUILTIN 2
#endif

const char* ssid = "SSID";  // WiFi SSID
const char* password = "PASSWORD";  // WiFi Password
const char* mqtt_server = "192.168.1.107";  // MQTT broker address
const char* mqtt_topic_full = "esp32/bme680_full";
const char* mqtt_topic_temp = "esp32/bme680_temp";

WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_BME680 bme;

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting with ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.print("WiFi Connected: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting with MQTT...");
    if (client.connect("ESP32_BME680_Client")) {
      Serial.println("Connected!");
    } else {
      Serial.print("Error, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5s");
      delay(5000);
    }
  }
}

// Simple function for converting resistance to “IAQ” 0–500
float calculateIAQ(float gas_kohm) {
  if (gas_kohm > 50) return 50;        // very good air
  else if (gas_kohm > 10) return 150;  // good
  else if (gas_kohm > 5) return 250;   // moderate
  else if (gas_kohm > 1) return 350;   // bad
  else return 450;                     // very bad
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  Wire.begin();

  if (!bme.begin()) {
    Serial.println("Sensor BME680 not found!");
    while (1);
  }

  setup_wifi();
  client.setServer(mqtt_server, 1883);

  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme.setGasHeater(320, 150);
}

void loop() {
  digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));

  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  if (!bme.performReading()) {
    Serial.println("Reading error!");
    return;
  }

  float temp = bme.temperature;
  float hum = bme.humidity;
  float pres = bme.pressure / 100.0;
  float gas = bme.gas_resistance / 1000.0; // kΩ
  float iaq = calculateIAQ(gas);

  Serial.printf("Temp: %.2f °C | Hum: %.2f %% | Pres: %.2f hPa | Gas: %.2f kΩ | AQI: %.0f\n",
                temp, hum, pres, gas, iaq);

  char payload[200];
  snprintf(payload, sizeof(payload),
           "{\"temperature\": %.2f, \"humidity\": %.2f, \"pressure\": %.2f, \"gas\": %.2f, \"iaq\": %.0f}",
           temp, hum, pres, gas, iaq);

  client.publish(mqtt_topic_full, payload);
  
  char tempStr[16];
  dtostrf(temp, 4, 2, tempStr);
  client.publish(mqtt_topic_temp, tempStr);

  delay(8000);
}
