# HairCLIP: Design Your Hair by Text and Reference Image
<pre>
Implemented the hairclip inference.
* https://github.com/wty-ustc/HairCLIP
* https://huggingface.co/spaces/Gradio-Blocks/HairCLIP/tree/main
* https://huggingface.co/spaces/Gradio-Blocks/HairCLIP
</pre>

## Working:
<pre>
1. Dedect / align face . (align.ipynb)
2. convert face to latent vector (e4e_preojection.py)
3. Use Face latent vector to apply hairclip model weights . (RunClip.py)
4. Apply U2net
</pre>

<hr> 

## FullBody
<pre>
1. Find the box cordinate from face align.
2. Run the inference on the face. (i.e hair clip image)
3. place the new image on top of the original image .
</pre>

### U2Net 
* https://github.com/xuebinqin/U-2-Net
<pre>
1. Use u2net to find the segement the image/face . 
2.  mount the face mask on top of the image. 
</pre>