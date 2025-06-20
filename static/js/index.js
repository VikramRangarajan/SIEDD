/**
 * SIEDD Cropper Carousel Logic
 * Replaces bulmaCarousel with synchronized cropper carousel.
 * This script expects the HTML structure from the new carousel section in index.html.
 */

const PREVIEW_W = 400, PREVIEW_H = 240;
// === Your image sets here ===
const slides = [
  {
    ground: 'static/images/jockey_hd_viz/gt_jockey_240.png',
    groundCaption: 'Ground Truth (Jockey)',
    others: [
      'static/images/jockey_hd_viz/gt_jockey_240.png',
      'static/images/jockey_hd_viz/siedd_jockey_240.png',
      'static/images/jockey_hd_viz/hnerv_jockey_240.png',
      'static/images/jockey_hd_viz/hinerv_jockey_240.png',
      'static/images/jockey_hd_viz/ffnerv_jockey_240.png'
    ],
    previewCaptions: [
      'Ground Truth',
      'SIEDD',
      'HNeRV',
      'HiNeRV',
      'FFNeRV'
    ],
    carouselCaption: 'Reconstruction results of SIEDD vs. HiNeRV, HNeRV, and FFNeRV. The images shown are from UVG-HD Jockey. All models were given a maximum of 1 hour of training time.'
  },
  {
    ground: 'static/images/bosphorus_hd_viz/bos_0.png',
    groundCaption: 'Ground Truth (Bosphorus)',
    others: [
      'static/images/bosphorus_hd_viz/bos_0.png',
      'static/images/bosphorus_hd_viz/siedd_bos_0.png',
      'static/images/bosphorus_hd_viz/hnerv_bos_0.png',
      'static/images/bosphorus_hd_viz/hinerv_bos_0.png',
      'static/images/bosphorus_hd_viz/ffnerv_bos_0.png'
    ],
    previewCaptions: [
      'Ground Truth',
      'SIEDD',
      'HNeRV',
      'HiNeRV',
      'FFNeRV'
    ],
    carouselCaption: 'Reconstruction results of SIEDD vs. HiNeRV, HNeRV, and FFNeRV. The images shown are from UVG-HD Bosphorus. All models were given a maximum of 1 hour of training time.'
  },
  {
    ground: 'static/images/beauty_super_resolution/Beauty_SR_gt.png',
    groundCaption: 'Ground Truth (UVG-4k frame)',
    others: [
      'static/images/beauty_super_resolution/Beauty_SR_gt.png',
      'static/images/beauty_super_resolution/Beauty_SR_SIEDD.png',
      'static/images/beauty_super_resolution/Beauty_SR_bilinear.png',
      'static/images/beauty_super_resolution/Beauty_SR_bicubic.png',
      'static/images/beauty_super_resolution/Beauty_SR_nearest.png'
    ],
    previewCaptions: [
      'Ground Truth',
      'SIEDD-L',
      'Bilinear',
      'Bicubic',
      'Nearest'
    ],
    carouselCaption: 'Super resolution results. SIEDD-L was trained on UVG-HD Beauty and set to decode at 4k. The results are compared with interpolation methods (nearest, bilinear, bicubic) and the ground truth UVG-4k frame.'
  },
  {
    ground: 'static/images/shakendry_4k/gt.png',
    groundCaption: 'Ground Truth (UVG-4k frame)',
    others: [
      'static/images/shakendry_4k/gt.png',
      'static/images/shakendry_4k/4k_1x1.png',
      'static/images/shakendry_4k/4k_3x3.png',
      'static/images/shakendry_4k/4k_6x6.png',
    ],
    previewCaptions: [
      'Ground Truth',
      'No Patching',
      '3x3 Patches',
      '6x6 Patches',
    ],
    carouselCaption: 'UVG-4k visual results. We compare the ground truth frame to SIEDD-L with no patching, 3x3 patches, and 6x6 patches. While patching provides much higher decoding speed, it causes a visible loss in quality. '
  },
  {
    ground: 'static/images/FLIP/yachtrideref.png',
    groundCaption: 'Ground Truth (UVG-HD frame)',
    others: [
      'static/images/FLIP/yachtrideref.png',
      'static/images/FLIP/yachtridepred.png',
      'static/images/FLIP/yachtrideflip.png',
    ],
    previewCaptions: [
      'Ground Truth',
      'SIEDD-L',
      'FLIP Error Map',
    ],
    carouselCaption: 'SIEDD-L reconstruction results on UVG-HD YachtRide. We also show the FLIP error map of the two frames. '
  },
  // Add more slides as needed
];

(function() {
  // Wait for DOMContentLoaded to ensure elements exist
  document.addEventListener('DOMContentLoaded', function() {
    const groundCanvas = document.getElementById('groundCanvas');
    if (!groundCanvas) return; // Only run if carousel is present

    const groundCtx = groundCanvas.getContext('2d');
    const previewGrid = document.getElementById('previewGrid');
    let previewCanvases = [];
    const slideIndicator = document.getElementById('slideIndicator');

    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');

    let currentSlide = 0;
    let dragging = false;
    let startX, startY, endX, endY;
    let groundImage = new window.Image();
    let variantImages = [];
    let dragMoved = false; // Track if mouse moved during drag

    function loadSlide(index) {
      const slide = slides[index];
      slideIndicator.textContent = `${index + 1} / ${slides.length}`;

      groundImage = new window.Image();
      variantImages = [];

      // Set ground truth caption
      const groundCaptionElem = document.getElementById('groundCaption');
      if (groundCaptionElem) {
        groundCaptionElem.textContent = slide.groundCaption || '';
      }

      // Set carousel-wide caption
      const carouselCaptionElem = document.getElementById('carouselCaption');
      if (carouselCaptionElem) {
        carouselCaptionElem.textContent = slide.carouselCaption || '';
      }

      groundImage.onload = () => {
        const maxW = 500, maxH = 300;
        const scale = Math.min(maxW / groundImage.width, maxH / groundImage.height, 1);
        const scaledW = groundImage.width * scale;
        const scaledH = groundImage.height * scale;

        groundCanvas.width = scaledW;
        groundCanvas.height = scaledH;
        groundCtx.drawImage(groundImage, 0, 0, scaledW, scaledH);
      };
      groundImage.src = slide.ground;

      // Remove old previews
      previewGrid.innerHTML = "";
      previewCanvases = [];
      previewLinks = [];

      // Load variants and captions
      let loadedCount = 0;
      slide.others.forEach((src, i) => {
        const img = new window.Image();
        img.onload = () => {
          loadedCount++;
          // When all variant images are loaded and not dragging, show full images in previews
          if (loadedCount === slide.others.length && !dragging) {
            showFullPreviews();
          }
          // If dragging, update previews as before
          if (dragging) updatePreviews();
        };
        img.src = src;
        variantImages[i] = img;

        // Create preview canvas and link
        const link = document.createElement('a');
        link.target = "_blank";
        link.download = "crop.png";
        const canvas = document.createElement('canvas');
        canvas.className = "previewCanvas";
        canvas.width = PREVIEW_W;
        canvas.height = PREVIEW_H;
        link.appendChild(canvas);

        // Create caption for this preview
        const fig = document.createElement('figure');
        fig.style.display = "flex";
        fig.style.flexDirection = "column";
        fig.style.alignItems = "center";
        fig.appendChild(link);

        const figcaption = document.createElement('figcaption');
        figcaption.className = "has-text-centered";
        figcaption.style.marginTop = "0.5em";
        figcaption.textContent = (slide.previewCaptions && slide.previewCaptions[i]) ? slide.previewCaptions[i] : '';
        fig.appendChild(figcaption);

        // If this is the last preview and the number is odd, center it by spanning both columns
        if (
          slide.others.length % 2 === 1 &&
          i === slide.others.length - 1
        ) {
          fig.style.gridColumn = "1 / span 2";
        }

        previewGrid.appendChild(fig);
        previewCanvases.push(canvas);
        previewLinks.push(link);
      });
    }

    // Show the whole image in each preview canvas and link to the full image
    function showFullPreviews() {
      previewCanvases.forEach((canvas, i) => {
        const ctx = canvas.getContext('2d');
        const img = variantImages[i];
        if (!img || !img.complete) return;
        // Always fixed preview size
        canvas.width = PREVIEW_W;
        canvas.height = PREVIEW_H;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.imageSmoothingEnabled = false;
        if (ctx.imageSmoothingQuality !== undefined) ctx.imageSmoothingQuality = 'low';
        const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
        const drawW = img.width * scale;
        const drawH = img.height * scale;
        const offsetX = (canvas.width - drawW) / 2;
        const offsetY = (canvas.height - drawH) / 2;
        ctx.drawImage(img, 0, 0, img.width, img.height, offsetX, offsetY, drawW, drawH);
      });
    }

    function updatePreviews() {
      const cropX = Math.min(startX, endX);
      const cropY = Math.min(startY, endY);
      const cropW = Math.abs(endX - startX);
      const cropH = Math.abs(endY - startY);

      // If crop area is zero, show full previews instead
      if (cropW === 0 || cropH === 0) {
        showFullPreviews();
        return;
      }

      previewCanvases.forEach((canvas, i) => {
        const ctx = canvas.getContext('2d');
        const img = variantImages[i];
        if (!img.complete) return;

        // Always fixed preview size
        canvas.width = PREVIEW_W;
        canvas.height = PREVIEW_H;
        ctx.clearRect(0, 0, PREVIEW_W, PREVIEW_H);

        const scaleUp = img.width / groundCanvas.width;
        const sx = cropX * scaleUp;
        const sy = cropY * scaleUp;
        const sw = cropW * scaleUp;
        const sh = cropH * scaleUp;

        // Draw the crop, stretched to fit preview canvas
        ctx.imageSmoothingEnabled = false;
        if (ctx.imageSmoothingQuality !== undefined) ctx.imageSmoothingQuality = 'low';
        ctx.drawImage(
          img,
          sx, sy, sw, sh,
          0, 0, PREVIEW_W, PREVIEW_H
        );
      });
    }

    function getEventPos(e) {
      const rect = groundCanvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return {
        x: clientX - rect.left,
        y: clientY - rect.top
      };
    }

    function startDrag(e) {
      e.preventDefault();
      const pos = getEventPos(e);
      startX = pos.x;
      startY = pos.y;
      dragging = true;
      dragMoved = false;
    }

    function moveDrag(e) {
      if (!dragging) return;
      e.preventDefault();
      const pos = getEventPos(e);
      endX = pos.x;
      endY = pos.y;

      // If mouse/touch moved more than a few pixels, consider it a drag
      if (Math.abs(endX - startX) > 3 || Math.abs(endY - startY) > 3) {
        dragMoved = true;
      }

      const width = endX - startX;
      const height = endY - startY;

      groundCtx.clearRect(0, 0, groundCanvas.width, groundCanvas.height);
      groundCtx.drawImage(groundImage, 0, 0, groundCanvas.width, groundCanvas.height);
      groundCtx.strokeStyle = 'red';
      groundCtx.lineWidth = 2;
      groundCtx.strokeRect(startX, startY, width, height);

      updatePreviews();
    }

    function endDrag(e) {
      if (!dragging) return;
      dragging = false;

      // If not moved (click), reset crop and show full previews
      if (!dragMoved) {
        startX = startY = endX = endY = undefined;
        showFullPreviews();
        // Also redraw ground image without crop rectangle
        groundCtx.clearRect(0, 0, groundCanvas.width, groundCanvas.height);
        groundCtx.drawImage(groundImage, 0, 0, groundCanvas.width, groundCanvas.height);
        return;
      }

      // If crop is zero, show full previews
      if (
        startX === undefined || startY === undefined ||
        endX === undefined || endY === undefined ||
        Math.abs(endX - startX) === 0 || Math.abs(endY - startY) === 0
      ) {
        showFullPreviews();
      }
    }

    groundCanvas.addEventListener('mousedown', startDrag);
    groundCanvas.addEventListener('mousemove', moveDrag);
    groundCanvas.addEventListener('mouseup', endDrag);

    groundCanvas.addEventListener('touchstart', startDrag);
    groundCanvas.addEventListener('touchmove', moveDrag);
    groundCanvas.addEventListener('touchend', endDrag);

    // Carousel navigation
    prevBtn.addEventListener('click', () => {
      currentSlide = (currentSlide - 1 + slides.length) % slides.length;
      loadSlide(currentSlide);
    });

    nextBtn.addEventListener('click', () => {
      currentSlide = (currentSlide + 1) % slides.length;
      loadSlide(currentSlide);
    });

    // Init
    loadSlide(currentSlide);
  });
})();
