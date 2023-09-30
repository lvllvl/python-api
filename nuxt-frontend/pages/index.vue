<template>
  <div class="min-h-screen flex flex-col items-center justify-center bg-gray-100">

  <!-- Hidden Image Input -->
  <input type="file" ref="imageInput" style="display: none;" value="../../data/image_portrait.png" />

    
    <!-- Title -->
    <h1 class="text-4xl font-bold mb-8">Parallel Image Processing</h1>
    
    <!-- Images -->
    <div class="flex">
      <!-- Preloaded Image -->
      <img src="../../data/image_portrait.png" alt="Preloaded Image" class="mr-4 w-64 h-64">

      
      <!-- Processed Image -->
      <img v-if="processedImage" :src="processedImage" alt="Processed Image">
    </div>
    
    <!-- Buttons -->
    <div class="mt-8">
      <p class="mb-4 text-lg font-semibold">Parallel image processing options:</p>
      <button @click="processImage('color-to-grayscale')" class="bg-blue-500 text-white py-2 px-4 rounded mr-4 hover:bg-blue-600">
        Color-to-grayscale
      </button>
      <button @click="processImage('image-blur')" class="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600">
        Image Blur
      </button>
      <button @click="processImage('reset-image')" class="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600">
        Reset 
      </button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      preloadedImageBase64; null,
      processedImage: null
    }
  },
  mounted() {
    // Convert the preloaded image to base64
    this.convertImageToBase64('../../data/image_portrait.png');
  },
  methods: {
  async convertImageToBase64(){
    imagePath = '../../data/image_portrait.png';
    let response = await fetch( imagePath );
    let blob = await response.blob();
    let reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onload = () => {
      this.preloadedImageBase64 = reader.result;
    };
    },
  async processImage(option) {
  try {
    let formData = new FormData();
    formData.append('imageData', this.preloadedImageBase64);
    formData.append('type', option);

    let response = await this.$axios.post('http://localhost:5000/process_image', formData);
    
    // Update the processedImage data property with the returned image path
    this.processedImage = URL.createObjectURL(response.data);

  } catch (error) {
    console.error("Error processing the image:", error);
  }
}

  }
}
</script>

<style scoped>
img {
    width: 500px;
    height: 400px;
}
</style>
