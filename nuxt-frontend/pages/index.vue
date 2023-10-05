<template>
  <div
    class="min-h-screen flex flex-col items-center justify-center bg-gray-100"
  >
    <!-- Image Upload -->
    <label for="imageUpload" class="text-lg font-semibold mb-4"
      >Upload your image:</label
    >
    <input
      type="file"
      id="imageUpload"
      ref="imageInput"
      @change="handleImageUpload"
    />

    <!-- Title -->
    <h1 class="text-4xl font-bold mb-8">Parallel Image Processing</h1>

    <!-- Images -->

    <div class="flex">
      <!-- Uploaded or Default Image -->
      <img
        :src="preloadedImageBase64 ? preloadedImageBase64 : defaultImage"
        alt="Uploaded or Default Image"
        class="mr-4 w-64 h-64"
      />

      <!-- Processed Image -->
      <img
        v-if="processedImage"
        :src="processedImage"
        alt="Processed Image"
        class="mr-4 w-64 h-64"
      />
    </div>

    <!-- Buttons -->
    <div class="mt-8">
      <p class="mb-4 text-lg font-semibold">
        Parallel image processing options:
      </p>
      <button
        @click="processImage('color-to-grayscale')"
        class="bg-blue-500 text-white py-2 px-4 rounded mr-4 hover:bg-blue-600"
      >
        Color-to-grayscale
      </button>
      <button
        @click="processImage('image-blur')"
        class="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
      >
        Image Blur
      </button>

      <button
        @click="testApi"
        class="bg-blue-500 text-white py-2 px-4 rounded mr-4 hover:bg-blue-600"
      >
        Test API
      </button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      defaultImage: "/data/image_portrait.png",
      preloadedImageBase64: null,
      processedImage: null,
    };
  },
  mounted() {
    // Convert the preloaded image to base64
    this.convertImageToBase64("../../data/image_portrait.png");
  },
  methods: {
    handleImageUpload(event) {
      let file = event.target.files[0];
      if (!file) return;

      // Check for file type
      if (!["image/jpeg", "image/png"].includes(file.type)) {
        alert("Only JPEG, PNG images are supported.");
        return;
      }

      // Check for file size (e.g., max 5MB)
      if (file.size > 5 * 1024 * 1024) {
        alert(
          "The uploaded image is too large. Please choose an image less than 5MB."
        );
        return;
      }

      let reader = new FileReader();
      reader.onload = (e) => {
        this.preloadedImageBase64 = e.target.result;
        alert("Image uploaded successfully!");
      };
      reader.onerror = () => {
        alert("There was an error uploading your image. Please try again.");
      };
      reader.readAsDataURL(file);
    },

    async testApi() {
      try {
        let response = await this.$axios.get("/test");
        console.log(response.data);
        alert( response.data.message);
      } catch (error) {
        console.error("Error calling the test API:", error);
      }
    },

    async convertImageToBase64() {
      let imagePath = "/data/image_portrait.png";
      let response = await fetch(imagePath);
      let blob = await response.blob();
      let reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onload = () => {
        this.preloadedImageBase64 = reader.result;
      };
    },
    async processImage(option) {
      try {
        let imageToProcess = this.preloadedImageBase64
          ? this.preloadedImageBase64
          : this.defaultImage;

        if (!imageToProcess) {
          alert("Please upload an image first.");
          return;
        }

        let formData = new FormData();
        formData.append("imageData", imageToProcess);
        formData.append("type", option);

        let response = await this.$axios.post(
          "http://localhost:5001/process_image",
          formData
        );

        // Update the processedImage data property with the returned image path
        this.processedImage = URL.createObjectURL(response.data);
        alert("Image processed successfully!");
      } catch (error) {
        console.error("Error processing the image:", error);
        alert("There was an error processing your image. Please try again.");
      }
    },
  },
};
</script>

<style scoped>
img {
  width: 500px;
  height: 400px;
}
</style>
