const $=s=>document.querySelector(s),messages=$('#messages'),input=$('#input'),form=$('#chatForm'),settings=$('#settings'),keyInput=$('#key'),logPanel=$('#logPanel'),logsEnabled=$('#logsEnabled');
keyInput.value=localStorage.getItem('accessKey')||'';logsEnabled.checked=localStorage.getItem('logsEnabled')==='true';
function setLogs(on){localStorage.setItem('logsEnabled',String(on));logPanel.classList.toggle('enabled',on);logPanel.setAttribute('aria-hidden',String(!on));if(on)writeLog('Журнал включён','success')}
function writeLog(text,type=''){if(!logsEnabled.checked)return;const row=document.createElement('div'),time=document.createElement('time');row.className=`log-entry ${type}`;time.textContent=new Date().toLocaleTimeString('ru-RU');row.append(time,document.createTextNode(text));$('#logEntries').appendChild(row);$('#logEntries').scrollTop=$('#logEntries').scrollHeight}
setLogs(logsEnabled.checked);logsEnabled.onchange=()=>setLogs(logsEnabled.checked);$('#collapseLogs').onclick=()=>logPanel.classList.add('collapsed');$('#expandLogs').onclick=()=>logPanel.classList.remove('collapsed');$('#clearLogs').onclick=()=>{$('#logEntries').replaceChildren();writeLog('Журнал очищен')};
$('#settingsBtn').onclick=()=>settings.showModal();$('#toggle').onclick=()=>keyInput.type=keyInput.type==='password'?'text':'password';
$('#saveKey').onclick=async()=>{const status=$('#keyStatus'),key=keyInput.value.trim();status.className='status';status.textContent='Проверяю…';writeLog('Проверка ключа доступа');try{const r=await fetch('/api/auth',{method:'POST',headers:{'X-Access-Key':key}});if(!r.ok)throw new Error((await r.json()).detail);localStorage.setItem('accessKey',key);status.classList.add('ok');status.textContent='Ключ принят';writeLog('Доступ подтверждён','success');setTimeout(()=>settings.close(),500)}catch(e){status.classList.add('bad');status.textContent=e.message||'Не удалось проверить ключ';writeLog(`Ошибка доступа: ${e.message}`,'error')}};
document.querySelectorAll('.suggestions button').forEach(b=>b.onclick=()=>{input.value=b.textContent;input.focus()});input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,130)+'px'});input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});
function add(text,type,sources=[]){document.querySelector('.welcome')?.remove();const el=document.createElement('div');el.className=`message ${type}`;el.textContent=text;if(sources.length){const s=document.createElement('small');s.className='sources';s.textContent='Источники: '+sources.join(' · ');el.appendChild(s)}messages.appendChild(el);messages.scrollTop=messages.scrollHeight;return el}
form.onsubmit=async e=>{
  e.preventDefault();const text=input.value.trim(),key=localStorage.getItem('accessKey');if(!text)return;
  if(!key){writeLog('Нет сохранённого ключа','error');settings.showModal();return}
  add(text,'user');writeLog(`Вопрос отправлен (${text.length} симв.)`);input.value='';input.style.height='auto';
  const answer=add('Подбираю статьи…','bot typing');$('#sendBtn').disabled=true;const started=performance.now();let fullText='',sources=[];
  try{
    writeLog('Поиск фрагментов и запуск потоковой генерации');
    const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json','X-Access-Key':key},body:JSON.stringify({message:text})});
    if(!response.ok){const raw=await response.text();let detail=raw;try{detail=JSON.parse(raw).detail}catch{}throw new Error(detail||`Ошибка сервера (HTTP ${response.status})`)}
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    while(true){
      const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const lines=buffer.split('\n');buffer=lines.pop();
      for(const line of lines){if(!line.trim())continue;const event=JSON.parse(line);if(event.type==='token'){fullText+=event.data;answer.textContent=fullText;answer.classList.remove('typing');messages.scrollTop=messages.scrollHeight}else if(event.type==='sources'){sources=event.data}else if(event.type==='error'){throw new Error(event.data)}}
    }
    if(!fullText)throw new Error('Модель не вернула текст');
    if(sources.length){const sourceEl=document.createElement('small');sourceEl.className='sources';sourceEl.textContent='Источники: '+sources.join(' · ');answer.appendChild(sourceEl)}
    writeLog(`Потоковый ответ завершён за ${((performance.now()-started)/1000).toFixed(1)} с`,'success');if(sources.length)writeLog(`Источники: ${sources.join(' · ')}`);
  }catch(err){answer.textContent=err.message+(err.message.includes('ключ')?' — откройте настройки.':'');answer.classList.remove('typing');writeLog(`Ошибка запроса: ${err.message}`,'error')}
  finally{$('#sendBtn').disabled=false;input.focus()}
};
if(!localStorage.getItem('accessKey'))setTimeout(()=>settings.showModal(),350);
