import scipy.io.wavfile
import os

# Page Configuration
st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="centered")

st.title("🎵 AI-Powered Music Generator")
st.markdown("Generate music using Artificial Intelligence 🎶")

# Sidebar
st.sidebar.header("Music Settings")

genre = st.sidebar.selectbox(
    "Select Genre",
    ["Pop", "Classical", "EDM", "Jazz", "Rock", "Ambient"]
)

duration = st.sidebar.slider("Select Duration (seconds)", 5, 30, 10)

prompt = st.text_area("Describe the music you want to generate:")

if st.button("🎼 Generate Music"):

    full_prompt = f"{genre} music, {prompt}"

    with st.spinner("Generating music... Please wait ⏳"):

        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

        inputs = processor(
            text=[full_prompt],
            padding=True,
            return_tensors="pt",
        )

        audio_values = model.generate(**inputs, max_new_tokens=duration * 50)

        sampling_rate = model.config.audio_encoder.sampling_rate
        file_path = "generated_music.wav"

        scipy.io.wavfile.write(
            file_path,
            rate=sampling_rate,
            data=audio_values[0, 0].detach().cpu().numpy()
        )

        st.success("✅ Music Generated Successfully!")
        st.audio(file_path)

        with open(file_path, "rb") as file:
            st.download_button(
                label="⬇ Download Music",
                data=file,
                file_name="AI_generated_music.wav",
                mime="audio/wav"
            )
