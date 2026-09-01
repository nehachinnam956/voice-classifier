
import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, TextInput } from "react-native";
import { Audio } from "expo-av";
const DEFAULT_SERVER_URL =  "http://192.168.0.5:8000";
const RECORD_MS = 1000;

export default function App() {
  const [serverUrl, setServerUrl] = useState(DEFAULT_SERVER_URL);
  const [recording, setRecording] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | recording | uploading
  const [result, setResult] = useState(null); // { label, confidence }
  const [error, setError] = useState(null);

  async function startRecording() {
    const { status: perm } = await Audio.requestPermissionsAsync();
    if (perm !== "granted") {
      setError("Microphone permission denied.");
      return;
    }

    await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
    const { recording: rec } = await Audio.Recording.createAsync({
     android: {
        extension: ".m4a",
        outputFormat: Audio.AndroidOutputFormat.MPEG_4,
        audioEncoder: Audio.AndroidAudioEncoder.AAC,
        sampleRate: 16000,
        numberOfChannels: 1,
        bitRate: 128000,
      },
      ios: {
        extension: ".wav",
        audioQuality: Audio.IOSAudioQuality.HIGH,
        sampleRate: 16000,
        numberOfChannels: 1,
        bitRate: 128000,
        linearPCMBitDepth: 16,
        linearPCMIsBigEndian: false,
        linearPCMIsFloat: false,
      },
    });
    setRecording(rec);
    setStatus("recording");
    setResult(null);
    setError(null);

    setTimeout(() => stopRecordingAndClassify(rec), RECORD_MS);
  }

  async function stopRecordingAndClassify(rec) {
    setStatus("uploading");
    await rec.stopAndUnloadAsync();
    const uri = rec.getURI();

    try {
      const formData = new FormData();
      formData.append("file", {
        uri,
        name: "clip.m4a",
        type: "audio/m4a",
      });

      const response = await fetch(`${serverUrl}/predict`, {
        method: "POST",
        body: formData,
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (!response.ok) {
        const errBody = await response.text();
        throw new Error(`Server error ${response.status}: ${errBody}`);
      }

      const data = await response.json();
      setResult({ label: data.label, confidence: data.confidence });
    } catch (err) {
      console.error("Prediction request failed:", err);
      setError(
        `Couldn't reach the server. Check that inference_server.py is running ` +
        `and that your phone and laptop are on the same WiFi. (${err.message})`
      );
    } finally {
      setStatus("idle");
      setRecording(null);
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Voice Classifier</Text>

      <Text style={styles.label}>Server URL</Text>
      <TextInput
        style={styles.input}
        value={serverUrl}
        onChangeText={setServerUrl}
        placeholder="http://192.168.1.x:8000"
        autoCapitalize="none"
        autoCorrect={false}
      />

      <TouchableOpacity
        style={[styles.button, status !== "idle" && styles.buttonDisabled]}
        onPress={startRecording}
        disabled={status !== "idle"}
      >
        <Text style={styles.buttonText}>
          {status === "recording" ? "Listening..." : status === "uploading" ? "Processing..." : "Tap to speak"}
        </Text>
      </TouchableOpacity>

      {status === "uploading" && <ActivityIndicator style={{ marginTop: 20 }} />}

      {result && (
        <View style={styles.resultBox}>
          <Text style={styles.resultLabel}>{result.label}</Text>
          <Text style={styles.resultConfidence}>
            {(result.confidence * 100).toFixed(1)}% confidence
          </Text>
        </View>
      )}

      {error && <Text style={styles.errorText}>{error}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: "#fff", padding: 20 },
  title: { fontSize: 24, fontWeight: "600", marginBottom: 20 },
  label: { fontSize: 12, color: "#666", alignSelf: "flex-start", marginLeft: 4 },
  input: {
    width: "100%", borderWidth: 1, borderColor: "#ccc", borderRadius: 8,
    padding: 10, marginBottom: 24, fontSize: 14,
  },
  button: { backgroundColor: "#2563eb", paddingVertical: 18, paddingHorizontal: 36, borderRadius: 12 },
  buttonDisabled: { backgroundColor: "#93b4f0" },
  buttonText: { color: "#fff", fontSize: 18, fontWeight: "500" },
  resultBox: { marginTop: 40, alignItems: "center" },
  resultLabel: { fontSize: 32, fontWeight: "700", textTransform: "uppercase" },
  resultConfidence: { fontSize: 16, color: "#666", marginTop: 8 },
  errorText: { marginTop: 20, color: "#dc2626", textAlign: "center", fontSize: 13 },
});
