import { useEffect, useRef, useState } from 'react';
import type { GeobodyVoxelPreview, ProspectVoxelPreview } from '../../api/backend';

export default function ProspectVoxelCanvas({ preview }: { preview: ProspectVoxelPreview | GeobodyVoxelPreview }) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const [angles, setAngles] = useState({ yaw: 0.72, pitch: 0.62 });
    const drag = useRef<{x:number;y:number}|null>(null);
    useEffect(() => {
        const canvas = canvasRef.current; if (!canvas) return;
        const ctx = canvas.getContext('2d'); if (!ctx) return;
        const width = canvas.clientWidth, height = canvas.clientHeight, dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.max(1, Math.floor(width*dpr)); canvas.height = Math.max(1, Math.floor(height*dpr)); ctx.setTransform(dpr,0,0,dpr,0,0);
        ctx.clearRect(0,0,width,height); ctx.fillStyle = '#101923'; ctx.fillRect(0,0,width,height);
        const pts = preview.coordinates; if (!pts.length) return;
        const mins = preview.bounds_ijk.min, maxs = preview.bounds_ijk.max;
        const center = mins.map((v,i)=>(v+maxs[i])/2);
        const span = Math.max(...mins.map((v,i)=>maxs[i]-v),1);
        const scale = Math.min(width,height)*0.72/span;
        const cy=Math.cos(angles.yaw), sy=Math.sin(angles.yaw), cp=Math.cos(angles.pitch), sp=Math.sin(angles.pitch);
        const projected = pts.map(p=>{
            const x=(p[0]-center[0]), y=(p[1]-center[1]), z=(p[2]-center[2]);
            const rx=x*cy-y*sy, ry=x*sy+y*cy;
            const py=ry*cp-z*sp, depth=ry*sp+z*cp;
            return {x:width/2+rx*scale,y:height/2-py*scale,depth,z:p[2]};
        }).sort((a,b)=>a.depth-b.depth);
        const size=Math.max(1.4,Math.min(4.2,scale*0.72));
        for(const p of projected){
            const t=(p.z-mins[2])/Math.max(1,maxs[2]-mins[2]);
            const hue=205-190*t; ctx.fillStyle=`hsl(${hue},72%,${48+12*t}%)`;
            ctx.beginPath(); ctx.moveTo(p.x,p.y-size);ctx.lineTo(p.x+size,p.y);ctx.lineTo(p.x,p.y+size);ctx.lineTo(p.x-size,p.y);ctx.closePath();ctx.fill();
        }
        ctx.fillStyle='rgba(230,238,246,.8)';ctx.font='11px sans-serif';ctx.fillText('I / J / K voxel coordinates · drag to rotate',12,20);
    },[preview,angles]);
    return <div className="prospect-voxel-viewer">
        <div className="prospect-voxel-toolbar"><span><strong>{'prospect_id' in preview ? preview.prospect_id : `GB ${preview.geobody_id}`}</strong> · {preview.voxel_count.toLocaleString()} voxels</span><span>{preview.is_sampled?`Displaying ${preview.preview_count.toLocaleString()} sampled cells (1:${preview.sampling_stride})`:`Displaying all ${preview.preview_count.toLocaleString()} cells`}</span><button type="button" onClick={()=>setAngles({yaw:0.72,pitch:0.62})}>Reset view</button></div>
        <canvas ref={canvasRef} className="prospect-voxel-canvas" onPointerDown={e=>{drag.current={x:e.clientX,y:e.clientY};e.currentTarget.setPointerCapture(e.pointerId);}} onPointerMove={e=>{if(!drag.current)return;const dx=e.clientX-drag.current.x,dy=e.clientY-drag.current.y;drag.current={x:e.clientX,y:e.clientY};setAngles(a=>({yaw:a.yaw+dx*0.008,pitch:Math.max(-1.45,Math.min(1.45,a.pitch+dy*0.008))}));}} onPointerUp={()=>{drag.current=null;}} onPointerCancel={()=>{drag.current=null;}} />
        <div className="prospect-voxel-meta"><span>I: {preview.bounds_ijk.min[0]}–{preview.bounds_ijk.max[0]}</span><span>J: {preview.bounds_ijk.min[1]}–{preview.bounds_ijk.max[1]}</span><span>K: {preview.bounds_ijk.min[2]}–{preview.bounds_ijk.max[2]}</span></div>
        <small className="muted">This is a direct rendering of stored IJK cells. It is not georeferenced to IL/XL/TWT or physical distance in this view.</small>
    </div>;
}

