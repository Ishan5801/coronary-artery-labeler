(function(){
  const cache=new Map();
  const originalRead=window.readNiftiVolume;
  window.readNiftiVolume=async function(file){
    const key=`${file.name}|${file.size}|${file.lastModified}`;
    if(cache.has(key)) return cache.get(key);
    const pending=originalRead(file); cache.set(key,pending);
    try{return await pending}catch(e){cache.delete(key);throw e}
  };
  window.renderVtkVolumes=function(items,wrap,empty){
    try{
      if(!window.vtk||!vtk.Rendering?.Misc){renderVoxelCloud(items[items.length-1].volume,wrap,empty);return}
      let host=wrap.querySelector('.vtk-host');
      if(!host){wrap.querySelectorAll('canvas').forEach(c=>c.hidden=true);host=document.createElement('div');host.className='vtk-host';wrap.append(host)}
      let grw=volumeViews.get(host),renderer,rw;const firstShow=!grw;
      if(grw){renderer=grw.getRenderer();try{renderer.removeAllVolumes?.()}catch(e){}try{(host.__actors||[]).forEach(a=>a.delete?.())}catch(e){}host.__actors=[];rw=grw.getRenderWindow()}else{grw=vtk.Rendering.Misc.vtkGenericRenderWindow.newInstance({background:[.93,.96,.95]});grw.setContainer(host);grw.resize();renderer=grw.getRenderer();rw=grw.getRenderWindow();volumeViews.set(host,grw)}
      if(!host.__vtkResizeObserver){host.__vtkResizeObserver=new ResizeObserver(()=>{const view=volumeViews.get(host);if(view){view.resize();view.getRenderWindow().render()}});host.__vtkResizeObserver.observe(host)}
      const cam=renderer.getActiveCamera?.();const savedCam=(!firstShow&&cam)?{pos:cam.getPosition(),focal:cam.getFocalPoint(),up:cam.getViewUp(),angle:cam.getViewAngle?.()}:null;
      const actors=items.map(item=>makeVolumeActor(item.volume,item.color,item.opacity));host.__actors=actors;actors.forEach(actor=>renderer.addVolume(actor));enableProgressiveVolumeQuality(grw,actors.map(actor=>actor.getMapper()),displayVolume(items[0].volume).spacing);if(savedCam&&cam){cam.setPosition(...savedCam.pos);cam.setFocalPoint(...savedCam.focal);cam.setViewUp(...savedCam.up);if(savedCam.angle&&cam.setViewAngle)cam.setViewAngle(savedCam.angle)}else{renderer.resetCamera();const camera=renderer.getActiveCamera?.();if(camera)camera.zoom(1.45)}renderer.resetCameraClippingRange?.();rw.render();empty.style.display='none';
    }catch(e){console.warn('3D redraw failed',e);showMaskError(empty,'3D rendering failed. Select the branch again.')}
  };
  const originalRenderBranches=window.renderBranches;
  function renderQuickPicker(){const list=document.getElementById('quickList'),count=document.getElementById('quickCount');if(!list)return;count.textContent=branches.length; if(!branches.length){list.innerHTML='<div class="quick-empty">Upload branch masks to select them here.</div>';return}list.innerHTML=branches.map((f,i)=>{const id=i+'-'+f.name,present=window.getBranchLabelState?window.getBranchLabelState(id).present:(labels[id]===true?'Yes':labels[id]===false?'No':''),sel=i===window.selectedBranchIndex;return `<div class="quick-row ${sel?'selected-quick':''}"><button class="quick-select" data-index="${i}"><span class="quick-dot"></span><span class="quick-index">${String(i+1).padStart(2,'0')}</span><span class="quick-name" title="${f.name}">${f.name}</span></button><div class="quick-labels"><button class="quick-yes ${present==='Yes'?'active-yes':''}" data-index="${i}" data-label="yes">Yes</button><button class="quick-no ${present==='No'?'active-no':''}" data-index="${i}" data-label="no">No</button><button class="quick-unsure ${present==='Unsure'?'active-unsure':''}" data-index="${i}" data-label="unsure">Unsure</button></div>${window.quickDetailHTML?window.quickDetailHTML(id):''}</div>`}).join('');list.querySelectorAll('.quick-select').forEach(b=>b.onclick=()=>selectBranch(branches[+b.dataset.index],+b.dataset.index));list.querySelectorAll('.quick-labels button').forEach(b=>b.onclick=()=>{const i=+b.dataset.index,id=i+'-'+branches[i].name,next=b.dataset.label==='yes'?'Yes':b.dataset.label==='no'?'No':'Unsure';if(window.setBranchPresence)window.setBranchPresence(id,next);else{labels[id]=b.dataset.label==='yes';renderBranches()}})}
  window.arteryFilter='all';
  function moveBranchControls(){const side=document.querySelector('.branch-side');if(!side||document.getElementById('branchActions'))return;const actions=document.createElement('div');actions.id='branchActions';actions.className='branch-actions';const progress=document.querySelector('.branches .progress-wrap'),exportBtn=document.getElementById('exportBtn'),exportStatus=document.getElementById('exportStatus'),finishBtn=document.getElementById('finishBtn');if(progress)actions.append(progress);const buttons=document.createElement('div');buttons.className='branch-action-buttons';if(exportBtn)buttons.append(exportBtn);if(exportStatus)buttons.append(exportStatus);if(finishBtn)buttons.append(finishBtn);actions.append(buttons);side.append(actions)}
  function applyArteryFilter(){document.querySelectorAll('#quickList .quick-row').forEach(row=>{const name=row.querySelector('.quick-name')?.textContent||'';const side=/_l\.nii(?:\.gz)?$/i.test(name)?'left':/_r\.nii(?:\.gz)?$/i.test(name)?'right':'unsplit';row.hidden=window.arteryFilter!=='all'&&side!==window.arteryFilter})}
  function renderArteryFilter(){const panel=document.querySelector('.branch-side .panel-head');if(!panel)return;let status=document.getElementById('branchStatus');if(!status){status=document.createElement('span');status.id='branchStatus';status.className='branch-status';panel.after(status)}let bar=document.getElementById('arteryFilter');if(!bar){bar=document.createElement('div');bar.id='arteryFilter';bar.className='artery-filter';status.after(bar)}bar.innerHTML=['all','left','right'].map(side=>`<button type="button" class="artery-filter-btn ${window.arteryFilter===side?'active':''}" data-side="${side}">${side==='all'?'All arteries':side==='left'?'Left artery':'Right artery'}</button>`).join('');bar.querySelectorAll('button').forEach(button=>button.onclick=()=>{window.arteryFilter=button.dataset.side;renderArteryFilter();applyArteryFilter()});applyArteryFilter()}
  window.renderBranches=function(){originalRenderBranches();renderQuickPicker();renderArteryFilter()};
  function refreshMappedTags(){document.querySelectorAll('#branchTable .branch-row').forEach((row,i)=>{const tag=row.querySelector('.location');if(tag)tag.textContent=wholeFile&&i===window.selectedBranchIndex?'Mapped to full mask':'Awaiting full mask'})}
  window.selectedBranchIndex=-1;
  const originalSelectBranch=window.selectBranch;
  window.selectBranch=function(file,index){window.selectedBranchIndex=index;originalSelectBranch(file,index);refreshMappedTags();renderQuickPicker();renderArteryFilter()};
  const renderWithMapping=window.renderBranches;
  window.renderBranches=function(){renderWithMapping();refreshMappedTags()};
  window.renderQuickPicker=renderQuickPicker;window.renderArteryFilter=renderArteryFilter;window.applyArteryFilter=applyArteryFilter;
  renderQuickPicker();renderArteryFilter();moveBranchControls();
})();
