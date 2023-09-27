<!-- pages/index.vue -->
<template>
  <SidebarLayout pageTitle="Parallel Image Processing">
    <!-- Sidebar content goes here -->
    <template v-slot:sidebarContent>
      <div>
        <p style="color: red;">Here is some text, it's 816 a person</p>
      </div>
    </template>
    
    <!-- Page content goes here -->
    <div>
      <!-- Your page content -->
    </div>
  </SidebarLayout>
</template>

<script>
import SidebarLayout from '~/layouts/sidebar-layout.vue';

export default {
  components: {
    SidebarLayout,
  },
}
</script>
